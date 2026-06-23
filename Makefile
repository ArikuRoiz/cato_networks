.PHONY: test lint eval up down logs seed demo dev trace bot prod prod-demo prod-seed prod-logs prod-down build

TRADE ?= ""

# ---------------------------------------------------------------------------
# Local dev (no Docker — uses .venv directly)
# ---------------------------------------------------------------------------

test:
	uv run pytest tests/ -q --cov=src/firm

lint:
	uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/

eval:
	uv run python -m eval.replay --window data/windows/default.yaml

seed:
	uv run python -m firm.cli seed

demo:
	uv run python -m firm.cli demo

dev:
	uv run python -m firm.cli dev

trace:
	uv run python -m firm.cli trace --trade-id $(TRADE)

web:
	uv run python -m firm.cli web

bot:
	uv run python -m firm.cli bot

# ---------------------------------------------------------------------------
# Docker infra only (Postgres + Langfuse, no app container)
# ---------------------------------------------------------------------------

up:
	docker-compose up -d postgres langfuse
	@echo "Waiting for Postgres to be ready..."
	@docker-compose exec postgres sh -c 'until pg_isready -U firm -d firm; do sleep 1; done'
	@echo "Postgres ready."

down:
	docker-compose down

logs:
	docker-compose logs -f

# ---------------------------------------------------------------------------
# Production — full stack in Docker (build image + run everything)
# ---------------------------------------------------------------------------

build:
	docker-compose build firm-app

prod: build
	docker-compose up -d
	@echo "Waiting for Postgres to be ready..."
	@docker-compose exec postgres sh -c 'until pg_isready -U firm -d firm; do sleep 1; done'
	@echo "Running migrations + corpus seed..."
	docker-compose run --rm firm-app firm seed
	@echo "Starting firm-app in background..."
	docker-compose up -d firm-app
	@echo "Done. Tail logs with: make prod-logs"

prod-demo: build
	docker-compose up -d postgres langfuse
	@echo "Waiting for Postgres to be ready..."
	@docker-compose exec postgres sh -c 'until pg_isready -U firm -d firm; do sleep 1; done'
	docker-compose run --rm firm-app firm demo

prod-seed: build
	docker-compose up -d postgres
	@docker-compose exec postgres sh -c 'until pg_isready -U firm -d firm; do sleep 1; done'
	docker-compose run --rm firm-app firm seed

prod-logs:
	docker-compose logs -f firm-app

prod-down:
	docker-compose down -v
