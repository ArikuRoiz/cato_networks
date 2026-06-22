.PHONY: test lint eval up seed demo dev trace

TRADE ?= ""

test:
	pytest tests/ -q --cov=src/firm

lint:
	ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/

eval:
	python -m eval.replay --window data/windows/default.yaml

up:
	docker-compose up -d

seed:
	python -m firm.cli seed

demo:
	python -m firm.cli demo

dev:
	python -m firm.cli dev

trace:
	python -m firm.cli trace --trade-id $(TRADE)
