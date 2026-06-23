"""``firm seed`` — run migrations, verify frozen bars, embed the news corpus."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from firm.cli.output import _emit, _load_settings, _project_root


def _cmd_seed(args: argparse.Namespace) -> None:
    """Run migrations, load frozen bar CSVs, embed news corpus."""
    settings = _load_settings()
    root = _project_root()

    _emit({"step": "migrations", "status": "starting"})
    _run_migrations(settings.database_url)
    _emit({"step": "migrations", "status": "ok"})

    _check_bars(root / "data" / "bars")

    corpus_path = root / "data" / "news" / "corpus.json"
    _emit({"step": "corpus", "status": "starting", "path": str(corpus_path)})
    if not corpus_path.exists():
        _emit(
            {
                "step": "corpus",
                "status": "warning",
                "message": "corpus.json not found; skipping embedding",
            }
        )
    else:
        count = _embed_corpus(corpus_path, settings.database_url)
        _emit({"step": "corpus", "status": "ok", "articles_embedded": count})

    _emit({"step": "seed", "status": "done"})


def _check_bars(bars_dir: Path) -> None:
    """Emit a status event for the frozen bar CSV files."""
    _emit({"step": "bars", "status": "checking", "dir": str(bars_dir)})
    bar_files = list(bars_dir.glob("*.csv"))
    if not bar_files:
        _emit({"step": "bars", "status": "warning", "message": "no CSV files found in data/bars/"})
    else:
        _emit({"step": "bars", "status": "ok", "files": [f.name for f in bar_files]})


def _run_migrations(database_url: str) -> None:
    """Run Alembic migrations programmatically."""
    from alembic import command  # deferred: heavy import
    from alembic.config import Config

    from firm.persistence.db_url import to_sqlalchemy_url

    root = _project_root()
    alembic_cfg = Config(str(root / "migrations" / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(root / "migrations"))
    alembic_cfg.set_main_option("sqlalchemy.url", to_sqlalchemy_url(database_url))
    command.upgrade(alembic_cfg, "head")


def _embed_corpus(corpus_path: Path, database_url: str) -> int:
    """Parse corpus.json and upsert articles into pgvector via PgvectorEvidenceStore.

    Requires a live Postgres connection identified by *database_url*.
    Returns the number of articles processed.
    """
    import psycopg  # deferred: heavy import

    from firm.adapters.evidence_pgvector import PgvectorEvidenceStore
    from firm.orchestration.checkpointer import _normalise_database_url

    docs = _parse_news_docs(corpus_path)
    url = _normalise_database_url(database_url)
    with psycopg.connect(url) as conn:
        store = PgvectorEvidenceStore(conn)
        store.migrate()
        for doc in docs:
            store.embed_and_store(doc)
        conn.commit()
    return len(docs)


def _parse_news_docs(corpus_path: Path) -> list[Any]:
    """Read corpus.json and return a list of NewsDoc objects."""
    import json as _json

    from firm.ports.types import NewsDoc

    raw: list[dict[str, Any]] = _json.loads(corpus_path.read_text(encoding="utf-8"))
    return [
        NewsDoc(
            symbol=item["symbol"],
            text=item["text"],
            source_url=item["source_url"],
            published_at=datetime.fromisoformat(item["published_at"].replace("Z", "+00:00")),
        )
        for item in raw
    ]
