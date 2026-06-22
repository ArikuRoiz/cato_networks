# Terraform deployment view — The AI Investment Firm
#
# This file documents the path from docker-compose to AWS production.
# It is a VIEW ARTIFACT — written but NOT applied.
# Applying it requires real AWS credentials and incurs cost.
#
# Mapping:
#   firm-app container     → aws_ecs_service + aws_ecs_task_definition (Fargate)
#   postgres container     → aws_db_instance (RDS Postgres 16, Multi-AZ)
#   langfuse container     → aws_ecs_service (self-hosted on Fargate)
#   .env secrets           → aws_ssm_parameter (DATABASE_URL, ANTHROPIC_API_KEY, SLACK_BOT_TOKEN)

terraform {
  required_version = ">= 1.8"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state — replace with your own backend before applying.
  # backend "s3" {
  #   bucket = "your-tfstate-bucket"
  #   key    = "the-ai-firm/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# Networking (minimal — references existing VPC/subnets via variables)
# ---------------------------------------------------------------------------

data "aws_vpc" "main" {
  id = var.vpc_id
}

data "aws_subnets" "private" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.main.id]
  }
  filter {
    name   = "tag:Tier"
    values = ["private"]
  }
}

data "aws_subnets" "public" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.main.id]
  }
  filter {
    name   = "tag:Tier"
    values = ["public"]
  }
}

# ---------------------------------------------------------------------------
# Secrets — SSM Parameter Store (SecureString)
# ---------------------------------------------------------------------------

resource "aws_ssm_parameter" "database_url" {
  name        = "/${var.app_name}/DATABASE_URL"
  type        = "SecureString"
  value       = "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.postgres.address}:5432/${var.db_name}"
  description = "Full connection string for the firm Postgres database"
  tags        = local.common_tags
}

resource "aws_ssm_parameter" "anthropic_api_key" {
  name        = "/${var.app_name}/ANTHROPIC_API_KEY"
  type        = "SecureString"
  value       = var.anthropic_api_key
  description = "Anthropic API key for LLM calls"
  tags        = local.common_tags
}

resource "aws_ssm_parameter" "slack_bot_token" {
  name        = "/${var.app_name}/SLACK_BOT_TOKEN"
  type        = "SecureString"
  value       = var.slack_bot_token
  description = "Slack bot token for HITL approval messages and daily reports"
  tags        = local.common_tags
}

resource "aws_ssm_parameter" "langfuse_public_key" {
  name        = "/${var.app_name}/LANGFUSE_PUBLIC_KEY"
  type        = "SecureString"
  value       = var.langfuse_public_key
  description = "Langfuse public key for observability"
  tags        = local.common_tags
}

resource "aws_ssm_parameter" "langfuse_secret_key" {
  name        = "/${var.app_name}/LANGFUSE_SECRET_KEY"
  type        = "SecureString"
  value       = var.langfuse_secret_key
  description = "Langfuse secret key for observability"
  tags        = local.common_tags
}

# ---------------------------------------------------------------------------
# RDS Postgres 16 with Multi-AZ (replaces the postgres container)
# ---------------------------------------------------------------------------

resource "aws_db_subnet_group" "postgres" {
  name        = "${var.app_name}-postgres"
  subnet_ids  = data.aws_subnets.private.ids
  description = "Subnet group for the firm Postgres instance"
  tags        = local.common_tags
}

resource "aws_security_group" "postgres" {
  name        = "${var.app_name}-postgres-sg"
  description = "Allow inbound Postgres from the ECS tasks"
  vpc_id      = data.aws_vpc.main.id

  ingress {
    description     = "Postgres from ECS"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

resource "aws_db_instance" "postgres" {
  identifier             = "${var.app_name}-postgres"
  engine                 = "postgres"
  engine_version         = "16"
  instance_class         = var.db_instance_class
  allocated_storage      = 20
  max_allocated_storage  = 100
  storage_encrypted      = true
  storage_type           = "gp3"

  db_name  = var.db_name
  username = var.db_username
  password = var.db_password  # injected from variable; use aws_secretsmanager_secret in prod

  # HA: Multi-AZ standby — this is the documented HA path for the single-node SPOF.
  multi_az = true

  db_subnet_group_name   = aws_db_subnet_group.postgres.name
  vpc_security_group_ids = [aws_security_group.postgres.id]

  # pgvector extension must be enabled via a parameter group
  parameter_group_name = aws_db_parameter_group.postgres.name

  backup_retention_period = 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"

  skip_final_snapshot       = false
  final_snapshot_identifier = "${var.app_name}-postgres-final"

  deletion_protection = true

  tags = local.common_tags
}

resource "aws_db_parameter_group" "postgres" {
  name        = "${var.app_name}-postgres-pg16"
  family      = "postgres16"
  description = "Postgres 16 parameter group enabling pgvector"

  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements"
  }

  tags = local.common_tags
}

# Note: pgvector is installed via a migration (CREATE EXTENSION IF NOT EXISTS vector;)
# run by Alembic on first startup. RDS Postgres 16 ships pgvector 0.7+ by default.

# ---------------------------------------------------------------------------
# ECS Cluster
# ---------------------------------------------------------------------------

resource "aws_ecs_cluster" "main" {
  name = var.app_name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = local.common_tags
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
  }
}

# ---------------------------------------------------------------------------
# IAM — ECS task execution role
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "ecs_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_execution" {
  name               = "${var.app_name}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume_role.json
  tags               = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ecs_execution_managed" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "ssm_read" {
  statement {
    actions   = ["ssm:GetParameters", "ssm:GetParameter"]
    resources = ["arn:aws:ssm:${var.aws_region}:${var.aws_account_id}:parameter/${var.app_name}/*"]
  }
}

resource "aws_iam_role_policy" "ecs_execution_ssm" {
  name   = "ssm-read"
  role   = aws_iam_role.ecs_execution.id
  policy = data.aws_iam_policy_document.ssm_read.json
}

resource "aws_iam_role" "ecs_task" {
  name               = "${var.app_name}-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume_role.json
  tags               = local.common_tags
}

# ---------------------------------------------------------------------------
# Security group for ECS tasks
# ---------------------------------------------------------------------------

resource "aws_security_group" "ecs_tasks" {
  name        = "${var.app_name}-ecs-tasks-sg"
  description = "Outbound-only security group for ECS tasks"
  vpc_id      = data.aws_vpc.main.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# firm-app — ECS Fargate service (replaces the firm-app container)
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "firm_app" {
  name              = "/ecs/${var.app_name}/firm-app"
  retention_in_days = 30
  tags              = local.common_tags
}

resource "aws_ecs_task_definition" "firm_app" {
  family                   = "${var.app_name}-firm-app"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "firm-app"
      image     = "${var.ecr_repository_url}:${var.app_image_tag}"
      essential = true

      secrets = [
        { name = "DATABASE_URL",        valueFrom = aws_ssm_parameter.database_url.arn },
        { name = "ANTHROPIC_API_KEY",   valueFrom = aws_ssm_parameter.anthropic_api_key.arn },
        { name = "SLACK_BOT_TOKEN",     valueFrom = aws_ssm_parameter.slack_bot_token.arn },
        { name = "LANGFUSE_PUBLIC_KEY", valueFrom = aws_ssm_parameter.langfuse_public_key.arn },
        { name = "LANGFUSE_SECRET_KEY", valueFrom = aws_ssm_parameter.langfuse_secret_key.arn },
      ]

      environment = [
        { name = "LANGFUSE_HOST", value = "http://langfuse.${var.app_name}.internal:3000" },
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.firm_app.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "firm-app"
        }
      }
    }
  ])

  tags = local.common_tags
}

resource "aws_ecs_service" "firm_app" {
  name            = "firm-app"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.firm_app.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.private.ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  # Ensure the DB is ready before the service starts
  depends_on = [aws_db_instance.postgres]

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# Langfuse — self-hosted ECS Fargate service (replaces the langfuse container)
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "langfuse" {
  name              = "/ecs/${var.app_name}/langfuse"
  retention_in_days = 14
  tags              = local.common_tags
}

resource "aws_ecs_task_definition" "langfuse" {
  family                   = "${var.app_name}-langfuse"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_execution.arn

  container_definitions = jsonencode([
    {
      name      = "langfuse"
      image     = "ghcr.io/langfuse/langfuse:latest"
      essential = true

      portMappings = [{ containerPort = 3000, protocol = "tcp" }]

      secrets = [
        { name = "DATABASE_URL",      valueFrom = aws_ssm_parameter.database_url.arn },
        { name = "LANGFUSE_SECRET_KEY", valueFrom = aws_ssm_parameter.langfuse_secret_key.arn },
      ]

      environment = [
        { name = "NEXTAUTH_URL",    value = "http://langfuse.${var.app_name}.internal:3000" },
        { name = "NEXTAUTH_SECRET", value = var.langfuse_nextauth_secret },
        { name = "SALT",            value = var.langfuse_salt },
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.langfuse.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "langfuse"
        }
      }
    }
  ])

  tags = local.common_tags
}

resource "aws_ecs_service" "langfuse" {
  name            = "langfuse"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.langfuse.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.private.ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  depends_on = [aws_db_instance.postgres]

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# Local values
# ---------------------------------------------------------------------------

locals {
  common_tags = {
    Project     = var.app_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "postgres_endpoint" {
  description = "RDS Postgres endpoint (hostname:port)"
  value       = "${aws_db_instance.postgres.address}:${aws_db_instance.postgres.port}"
  sensitive   = false
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "firm_app_task_definition_arn" {
  description = "Latest firm-app task definition ARN"
  value       = aws_ecs_task_definition.firm_app.arn
}
