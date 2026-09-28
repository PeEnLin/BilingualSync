"""
database/connection.py
~~~~~~~~~~~~~~~~~~~~~~
SQLite connection management for BilingualSync.

Provides:
* :class:`DatabaseConnection` – context-manager wrapper around a
  ``sqlite3.Connection`` with WAL mode, foreign-key enforcement, and
  automatic schema bootstrapping.
* :func:`get_connection` – factory that returns a ready-to-use
  ``DatabaseConnection`` from the application settings.

Threading model
---------------
Each call to :class:`DatabaseConnection` creates its own connection.
For multi-threaded use (e.g., Streamlit's rerun model) pass
``check_same_thread=False``; the WAL journal mode makes concurrent reads
safe while serialising writes through SQLite's built-in locking.

Usage::

    from database.connection import get_connection

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Optional, Type

from config.settings import Settings, get_settings
from core.exceptions import DatabaseError, SchemaError

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DDL – Schema Bootstrap SQL
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_VOCABULARY_TABLE = """
CREATE TABLE IF NOT EXISTS vocabulary (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    surface_form    TEXT    NOT NULL,
    base_form       TEXT    NOT NULL,
    reading         TEXT    NOT NULL DEFAULT '',
    part_of_speech  TEXT    NOT NULL DEFAULT '',
    source_lang     TEXT    NOT NULL DEFAULT 'ja',
    target_lang     TEXT    NOT NULL DEFAULT 'zh-tw',
    translation     TEXT    NOT NULL DEFAULT '',
    notes           TEXT    NOT NULL DEFAULT '',
    review_count    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    UNIQUE (surface_form, source_lang, target_lang)
);
"""

_CREATE_EXAMPLE_SENTENCES_TABLE = """
CREATE TABLE IF NOT EXISTS example_sentences (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    vocab_id         INTEGER NOT NULL REFERENCES vocabulary(id) ON DELETE CASCADE,
    source_text      TEXT    NOT NULL,
    target_text      TEXT    NOT NULL DEFAULT '',
    source_start_ms  INTEGER,
    source_end_ms    INTEGER,
    created_at       TEXT    NOT NULL
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_vocab_base_form   ON vocabulary (base_form);",
    "CREATE INDEX IF NOT EXISTS idx_vocab_source_lang ON vocabulary (source_lang);",
    "CREATE INDEX IF NOT EXISTS idx_example_vocab_id  ON example_sentences (vocab_id);",
]

_ALL_DDL: list[str] = [
    _CREATE_VOCABULARY_TABLE,
    _CREATE_EXAMPLE_SENTENCES_TABLE,
    *_CREATE_INDEXES,
]


# ─────────────────────────────────────────────────────────────────────────────
# DatabaseConnection
# ─────────────────────────────────────────────────────────────────────────────

class DatabaseConnection:
    """
    Context-manager wrapper around a ``sqlite3.Connection``.

    Opens a connection, configures pragmas, bootstraps the schema on first
    use, and guarantees the connection is closed on exit regardless of
    exceptions.

    Parameters
    ----------
    db_path:
        Path to the SQLite file. The file (and parent directories) will be
        created if they do not already exist.
    check_same_thread:
        Passed directly to ``sqlite3.connect``. Set ``False`` for
        multi-threaded consumers (e.g. Streamlit).

    Raises
    ------
    DatabaseError
        If the connection cannot be established.
    SchemaError
        If schema bootstrapping fails.

    Examples
    --------
    >>> with DatabaseConnection(Path("test.db")) as conn:
    ...     conn.execute("SELECT sqlite_version()").fetchone()
    """

    def __init__(
        self,
        db_path: Path,
        check_same_thread: bool = True,
    ) -> None:
        self._db_path = db_path
        self._check_same_thread = check_same_thread
        self._conn: Optional[sqlite3.Connection] = None

    # ── Context Manager Interface ─────────────────────────────────────────────

    def __enter__(self) -> sqlite3.Connection:
        """Open the connection, apply pragmas, and bootstrap the schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=self._check_same_thread,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"Failed to open SQLite database at {self._db_path}: {exc}",
                {"db_path": str(self._db_path)},
            ) from exc

        self._configure_pragmas()
        self._bootstrap_schema()
        logger.debug("Opened SQLite connection: %s", self._db_path)
        return self._conn

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """Commit or roll back, then close the connection."""
        if self._conn is None:
            return False

        if exc_type is None:
            self._conn.commit()
            logger.debug("Transaction committed.")
        else:
            self._conn.rollback()
            logger.warning(
                "Transaction rolled back due to %s: %s", exc_type.__name__, exc_val
            )

        self._conn.close()
        self._conn = None
        return False  # Do not suppress exceptions

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _configure_pragmas(self) -> None:
        """Apply SQLite PRAGMAs for performance and data integrity."""
        assert self._conn is not None
        pragmas = [
            "PRAGMA journal_mode = WAL;",       # Write-Ahead Logging for concurrency
            "PRAGMA foreign_keys = ON;",         # Enforce FK constraints
            "PRAGMA synchronous = NORMAL;",      # Balance between safety and speed
            "PRAGMA cache_size = -8192;",        # 8 MiB page cache
            "PRAGMA temp_store = MEMORY;",       # Keep temp tables in memory
        ]
        try:
            for pragma in pragmas:
                self._conn.execute(pragma)
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"Failed to configure SQLite PRAGMAs: {exc}"
            ) from exc

    def _bootstrap_schema(self) -> None:
        """Create tables and indexes if they do not already exist."""
        assert self._conn is not None
        try:
            for ddl in _ALL_DDL:
                self._conn.execute(ddl)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise SchemaError(
                f"Schema bootstrapping failed: {exc}"
            ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Public Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_connection(
    settings: Optional[Settings] = None,
    check_same_thread: bool = True,
) -> DatabaseConnection:
    """
    Create a :class:`DatabaseConnection` from the application settings.

    Parameters
    ----------
    settings:
        Optional :class:`~config.settings.Settings` instance. If ``None``,
        the module-level singleton returned by :func:`~config.settings.get_settings`
        is used.
    check_same_thread:
        Forwarded to :class:`DatabaseConnection`.

    Returns
    -------
    DatabaseConnection
        A context-manager ready to be used in a ``with`` statement.

    Examples
    --------
    >>> from database.connection import get_connection
    >>> with get_connection() as conn:
    ...     conn.execute("SELECT 1").fetchone()
    (1,)
    """
    resolved = settings or get_settings()
    return DatabaseConnection(
        db_path=resolved.db_path,
        check_same_thread=check_same_thread,
    )
