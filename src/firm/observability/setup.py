"""OpenTelemetry + Langfuse initialisation.

Call ``setup_telemetry()`` once at process start (before any spans are opened).
It wires a single ``TracerProvider`` that:

  1. Exports all spans to an OTLP collector (gRPC endpoint from env).
  2. Optionally adds the Langfuse span processor when ``LANGFUSE_PUBLIC_KEY``
     and ``LANGFUSE_SECRET_KEY`` are set — Langfuse 4.x integrates via OTel
     natively by adding its own ``LangfuseSpanProcessor`` to the provider we
     pass in.

The function is idempotent at the application level: calling it twice in tests
is safe because each call replaces the global provider, but callers should
avoid doing so in production.
"""

from __future__ import annotations

import logging
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

# Environment-variable names (kept as constants so callers can import them in tests)
_ENV_OTLP_ENDPOINT: str = "OTEL_EXPORTER_OTLP_ENDPOINT"
_ENV_LANGFUSE_PUBLIC_KEY: str = "LANGFUSE_PUBLIC_KEY"
_ENV_LANGFUSE_SECRET_KEY: str = "LANGFUSE_SECRET_KEY"

_DEFAULT_OTLP_ENDPOINT: str = "http://localhost:4317"


def _build_tracer_provider(service_name: str) -> TracerProvider:
    """Create a ``TracerProvider`` with service-name resource and OTLP exporter."""
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    endpoint = os.environ.get(_ENV_OTLP_ENDPOINT, _DEFAULT_OTLP_ENDPOINT)
    otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    return provider


def _maybe_add_langfuse(provider: TracerProvider) -> None:
    """Register a Langfuse span processor on *provider* when credentials exist.

    Langfuse 4.x integrates via OTel natively: passing ``tracer_provider`` to
    the ``Langfuse`` constructor causes it to call
    ``provider.add_span_processor(LangfuseSpanProcessor(...))`` internally.
    We do not import Langfuse at module level so the package remains optional.
    """
    public_key = os.environ.get(_ENV_LANGFUSE_PUBLIC_KEY)
    secret_key = os.environ.get(_ENV_LANGFUSE_SECRET_KEY)

    if not public_key or not secret_key:
        logger.debug("Langfuse credentials not found in env; Langfuse tracing disabled.")
        return

    try:
        from langfuse import Langfuse

        Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            tracer_provider=provider,
        )
        logger.info("Langfuse span processor registered on the global TracerProvider.")
    except ImportError:
        logger.warning("langfuse package not installed; Langfuse tracing will be skipped.")
    except Exception as exc:  # broad catch is deliberate for optional integration
        logger.warning("Failed to initialise Langfuse integration: %s", exc)


def setup_telemetry(service_name: str = "the-ai-firm") -> None:
    """Initialise the global OpenTelemetry ``TracerProvider``.

    Must be called once at process startup, before any agent or tool creates a
    span.  Subsequent calls replace the global provider (safe for tests, but
    avoid in production).

    Args:
        service_name: The ``service.name`` resource attribute attached to every
            span, visible in collector UIs and Langfuse project views.
    """
    provider = _build_tracer_provider(service_name)
    _maybe_add_langfuse(provider)
    trace.set_tracer_provider(provider)
    logger.info("OpenTelemetry TracerProvider initialised for service '%s'.", service_name)
