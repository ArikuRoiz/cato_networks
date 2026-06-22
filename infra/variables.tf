# Terraform variables — The AI Investment Firm deployment view
# Provide values via terraform.tfvars or environment variables (TF_VAR_*).
# This file is a VIEW ARTIFACT — written but NOT applied.

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "aws_account_id" {
  description = "AWS account ID — used to construct SSM ARNs"
  type        = string
  # No default — must be provided explicitly to avoid cross-account accidents.
}

variable "app_name" {
  description = "Application name prefix used for all resource names and tags"
  type        = string
  default     = "the-ai-firm"
}

variable "environment" {
  description = "Deployment environment (e.g. staging, production)"
  type        = string
  default     = "production"

  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "environment must be one of: staging, production"
  }
}

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

variable "vpc_id" {
  description = "ID of the VPC to deploy into (must have public and private subnets tagged Tier=public/private)"
  type        = string
  # No default — must be provided explicitly.
}

# ---------------------------------------------------------------------------
# RDS Postgres
# ---------------------------------------------------------------------------

variable "db_instance_class" {
  description = "RDS instance class for Postgres"
  type        = string
  default     = "db.t4g.small"
}

variable "db_name" {
  description = "Postgres database name"
  type        = string
  default     = "firm"
}

variable "db_username" {
  description = "Postgres master username"
  type        = string
  default     = "firm"
}

variable "db_password" {
  description = "Postgres master password — inject via TF_VAR_db_password or secrets manager"
  type        = string
  sensitive   = true
  # No default — must be provided explicitly and never committed.
}

# ---------------------------------------------------------------------------
# ECS / app image
# ---------------------------------------------------------------------------

variable "ecr_repository_url" {
  description = "ECR repository URL for the firm-app Docker image (e.g. 123456789.dkr.ecr.us-east-1.amazonaws.com/the-ai-firm)"
  type        = string
}

variable "app_image_tag" {
  description = "Docker image tag to deploy (e.g. git SHA or 'latest')"
  type        = string
  default     = "latest"
}

# ---------------------------------------------------------------------------
# Secrets — injected into SSM Parameter Store
# ---------------------------------------------------------------------------

variable "anthropic_api_key" {
  description = "Anthropic API key for LLM calls"
  type        = string
  sensitive   = true
}

variable "slack_bot_token" {
  description = "Slack bot token for HITL approval messages and daily reports"
  type        = string
  sensitive   = true
  default     = ""  # Optional — the fake Slack adapter is used when absent
}

variable "langfuse_public_key" {
  description = "Langfuse public key for observability"
  type        = string
  sensitive   = true
  default     = ""
}

variable "langfuse_secret_key" {
  description = "Langfuse secret key for observability"
  type        = string
  sensitive   = true
  default     = ""
}

variable "langfuse_nextauth_secret" {
  description = "NextAuth secret for the self-hosted Langfuse service"
  type        = string
  sensitive   = true
}

variable "langfuse_salt" {
  description = "Salt for the self-hosted Langfuse service"
  type        = string
  sensitive   = true
}
