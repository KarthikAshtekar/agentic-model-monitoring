"""Lifecycle-safe checkpointer construction for local agent workflows."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from monitoring_agent.paths import PROJECT_ROOT


@contextmanager
def create_checkpointer(
    backend: str,
    database_path: Path | None = None,
) -> Iterator[Any]:
    """Yield a memory or SQLite checkpointer and close durable resources cleanly."""
    if backend == "memory":
        yield InMemorySaver()
        return
    if backend != "sqlite":
        raise ValueError(
            f"Unsupported checkpoint backend {backend!r}; choose 'memory' or 'sqlite'."
        )
    if database_path is None:
        raise ValueError("A checkpoint database path is required for the SQLite backend.")

    resolved_path = (
        database_path
        if database_path.is_absolute()
        else PROJECT_ROOT / database_path
    )
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(resolved_path)) as checkpointer:
        checkpointer.setup()
        yield checkpointer
