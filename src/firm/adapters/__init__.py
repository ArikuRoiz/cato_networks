"""Concrete implementations of the port interfaces, plus in-memory fakes for testing."""

from firm.adapters.evidence_pgvector import PgvectorEvidenceStore
from firm.adapters.fakes import (
    FakeEvidenceStore,
    FakeLLM,
    FakeMarketData,
    FakeReportSink,
)
from firm.adapters.llm_anthropic import MODEL_MAP, AnthropicLLM
from firm.adapters.llm_cassette import CassetteLLM, CassetteNotFound
from firm.adapters.market_data_frozen import FrozenMarketData
from firm.adapters.market_data_live import LiveMarketData
from firm.adapters.report import ExcelReportSink, FileReportSink, SlackReportSink

__all__ = [
    "MODEL_MAP",
    "AnthropicLLM",
    "CassetteLLM",
    "CassetteNotFound",
    "ExcelReportSink",
    "FakeEvidenceStore",
    "FakeLLM",
    "FakeMarketData",
    "FakeReportSink",
    "FileReportSink",
    "FrozenMarketData",
    "LiveMarketData",
    "PgvectorEvidenceStore",
    "SlackReportSink",
]
