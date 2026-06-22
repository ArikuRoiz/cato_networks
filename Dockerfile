# The AI Investment Firm — production container image
#
# Multi-stage build:
#   builder  — installs dependencies into a virtual environment using uv
#   runtime  — copies only the venv + app source; no build tools in the final image
#
# Usage:
#   docker build -t the-ai-firm:latest .
#   docker run --env-file .env the-ai-firm:latest firm dev
#   docker run --env-file .env the-ai-firm:latest firm seed
#   docker run --env-file .env the-ai-firm:latest firm demo

# ---------------------------------------------------------------------------
# Stage 1 — dependency builder
# ---------------------------------------------------------------------------
FROM python:3.12.7-slim AS builder

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv==0.6.14

# Copy only the dependency manifests first to maximise layer-cache hits.
# Source code changes will not invalidate this layer.
COPY pyproject.toml uv.lock ./

# Sync production dependencies into .venv (no dev extras: pytest, mypy, ruff)
RUN uv sync --no-dev --frozen

# ---------------------------------------------------------------------------
# Stage 2 — runtime image
# ---------------------------------------------------------------------------
FROM python:3.12.7-slim AS runtime

LABEL org.opencontainers.image.title="The AI Investment Firm"
LABEL org.opencontainers.image.description="Multi-agent paper-trading desk — LangGraph + Anthropic + Postgres"
LABEL org.opencontainers.image.source="https://github.com/your-org/the-ai-firm"

# Create a non-root user for runtime security
RUN groupadd --gid 1001 firm && \
    useradd --uid 1001 --gid 1001 --no-create-home --shell /bin/false firm

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY src/ src/

# Copy frozen data and config (read-only at runtime; volumes can override)
COPY data/ data/
COPY config/ config/
COPY migrations/ migrations/
COPY eval/ eval/

# Activate the virtualenv for all subsequent RUN/CMD/ENTRYPOINT invocations
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure the app package is importable from src/ layout
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Transfer ownership of all /app contents to the non-root user so that
# eval/output/ and data/reports/ are writable at runtime.
RUN chown -R firm:firm /app

# Run as non-root
USER firm

# Health check — the app exposes no HTTP port; check the Python import instead
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import firm; print('ok')" || exit 1

# Default command — start the scheduler + event listener (background loop)
# Override at runtime: docker run ... firm seed / firm demo / firm trace --trade-id <id>
CMD ["firm", "dev"]
