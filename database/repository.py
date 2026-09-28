"""
database/repository.py
~~~~~~~~~~~~~~~~~~~~~~
Repository pattern implementation for the BilingualSync vocabulary store.

Provides :class:`VocabularyRepository` – the single gateway between the
application layer and the SQLite persistence layer.  All SQL lives here;
no other module should construct raw SQL strings.

Design highlights
-----------------
* **Repository Pattern**: hides all SQL behind a clean Python API.
* **Row ↔ Model mapping**: private ``_row_to_model`` helpers keep
  serialisation logic in one place.
* **Defensive programming**: every public method validates preconditions
  and raises domain exceptions (never raw ``sqlite3.Error``).
* **Fully typed**: all parameters and return values are annotated.

Usage::

    from database.connection import get_connection
    from database.repository import VocabularyRepository

    with get_connection() as conn:
        repo = VocabularyRepository(conn)
        entry = repo.add(VocabularyEntry(surface_form="食べた", base_form="食べる"))
        print(entry.id)
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from database.models import ExampleSentence, VocabularyEntry
from core.exceptions import (
    DatabaseError,
    DuplicateRecordError,
    RecordNotFoundError,
)

logger = logging.getLogger(__name__)

# ISO-8601 format used to persist datetimes as TEXT in SQLite
_DT_FORMAT = "%Y-%m-%dT%H:%M:%S.%f+00:00"


def _utcnow_str() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).strftime(_DT_FORMAT)


def _parse_dt(raw: str) -> datetime:
    """Parse an ISO-8601 UTC string back into a timezone-aware datetime."""
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        # Fallback: try without timezone suffix
        dt = datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S.%f")
        return dt.replace(tzinfo=timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# VocabularyRepository
# ─────────────────────────────────────────────────────────────────────────────

class VocabularyRepository:
    """
    CRUD repository for :class:`~database.models.VocabularyEntry` entities.

    All operations execute within the ``sqlite3.Connection`` supplied at
    construction time.  The caller is responsible for commit/rollback via the
    :class:`~database.connection.DatabaseConnection` context manager.

    Parameters
    ----------
    conn:
        An open ``sqlite3.Connection`` (obtained from
        :func:`~database.connection.get_connection`).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ── Private Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _row_to_vocab(row: sqlite3.Row | tuple) -> VocabularyEntry:
        """Convert a raw SQLite row tuple to a :class:`VocabularyEntry`."""
        (
            row_id,
            surface_form,
            base_form,
            reading,
            part_of_speech,
            source_lang,
            target_lang,
            translation,
            notes,
            review_count,
            created_at_str,
            updated_at_str,
        ) = row
        return VocabularyEntry(
            id=row_id,
            surface_form=surface_form,
            base_form=base_form,
            reading=reading,
            part_of_speech=part_of_speech,
            source_lang=source_lang,
            target_lang=target_lang,
            translation=translation,
            notes=notes,
            review_count=review_count,
            created_at=_parse_dt(created_at_str),
            updated_at=_parse_dt(updated_at_str),
        )

    @staticmethod
    def _row_to_example(row: tuple) -> ExampleSentence:
        """Convert a raw SQLite row tuple to an :class:`ExampleSentence`."""
        (
            row_id,
            vocab_id,
            source_text,
            target_text,
            source_start_ms,
            source_end_ms,
            created_at_str,
        ) = row
        return ExampleSentence(
            id=row_id,
            vocab_id=vocab_id,
            source_text=source_text,
            target_text=target_text,
            source_start_ms=source_start_ms,
            source_end_ms=source_end_ms,
            created_at=_parse_dt(created_at_str),
        )

    # ── Create ────────────────────────────────────────────────────────────────

    def add(self, entry: VocabularyEntry) -> VocabularyEntry:
        """
        Persist a new :class:`VocabularyEntry` and return it with its
        auto-generated ``id`` set.

        Parameters
        ----------
        entry:
            The transient (unsaved) vocabulary entry.  Its ``id`` field must
            be ``None``; use :meth:`update` for existing records.

        Returns
        -------
        VocabularyEntry
            A new immutable instance with ``id``, ``created_at``, and
            ``updated_at`` populated.

        Raises
        ------
        DuplicateRecordError
            If a record with the same ``(surface_form, source_lang, target_lang)``
            already exists.
        DatabaseError
            For any unexpected SQLite error.
        """
        now = _utcnow_str()
        sql = """
            INSERT INTO vocabulary
                (surface_form, base_form, reading, part_of_speech,
                 source_lang, target_lang, translation, notes,
                 review_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            entry.surface_form,
            entry.base_form,
            entry.reading,
            entry.part_of_speech,
            entry.source_lang,
            entry.target_lang,
            entry.translation,
            entry.notes,
            entry.review_count,
            now,
            now,
        )
        try:
            cursor = self._conn.execute(sql, params)
            new_id: int = cursor.lastrowid  # type: ignore[assignment]
        except sqlite3.IntegrityError as exc:
            raise DuplicateRecordError(
                "VocabularyEntry",
                "(surface_form, source_lang, target_lang)",
                f"{entry.surface_form}/{entry.source_lang}/{entry.target_lang}",
            ) from exc
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to insert VocabularyEntry: {exc}") from exc

        logger.debug("Inserted VocabularyEntry id=%d surface=%r", new_id, entry.surface_form)
        return entry.model_copy(
            update={
                "id": new_id,
                "created_at": _parse_dt(now),
                "updated_at": _parse_dt(now),
            }
        )

    def add_example(self, example: ExampleSentence) -> ExampleSentence:
        """
        Persist a new :class:`ExampleSentence` linked to an existing vocabulary
        entry and return it with ``id`` set.

        Parameters
        ----------
        example:
            Transient :class:`ExampleSentence` with a valid ``vocab_id``.

        Returns
        -------
        ExampleSentence
            Persisted instance with ``id`` populated.

        Raises
        ------
        RecordNotFoundError
            If ``example.vocab_id`` does not reference an existing vocabulary row.
        DatabaseError
            For any other SQLite error.
        """
        now = _utcnow_str()
        sql = """
            INSERT INTO example_sentences
                (vocab_id, source_text, target_text,
                 source_start_ms, source_end_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        try:
            cursor = self._conn.execute(
                sql,
                (
                    example.vocab_id,
                    example.source_text,
                    example.target_text,
                    example.source_start_ms,
                    example.source_end_ms,
                    now,
                ),
            )
            new_id: int = cursor.lastrowid  # type: ignore[assignment]
        except sqlite3.IntegrityError as exc:
            # FK violation means vocab_id does not exist
            raise RecordNotFoundError("VocabularyEntry", example.vocab_id) from exc
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to insert ExampleSentence: {exc}") from exc

        return example.model_copy(
            update={"id": new_id, "created_at": _parse_dt(now)}
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_by_id(self, vocab_id: int) -> VocabularyEntry:
        """
        Fetch a single :class:`VocabularyEntry` by primary key.

        Parameters
        ----------
        vocab_id:
            Primary key of the vocabulary entry.

        Returns
        -------
        VocabularyEntry
            The matching entry.

        Raises
        ------
        RecordNotFoundError
            If no row with that ``id`` exists.
        DatabaseError
            For unexpected SQLite errors.
        """
        sql = "SELECT * FROM vocabulary WHERE id = ?"
        try:
            row = self._conn.execute(sql, (vocab_id,)).fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to query VocabularyEntry by id: {exc}") from exc

        if row is None:
            raise RecordNotFoundError("VocabularyEntry", vocab_id)
        return self._row_to_vocab(row)

    def find_by_surface(
        self,
        surface_form: str,
        source_lang: str = "ja",
        target_lang: str = "zh-tw",
    ) -> Optional[VocabularyEntry]:
        """
        Look up a vocabulary entry by its exact surface form and language pair.

        Returns ``None`` rather than raising if the record does not exist,
        making it safe to use as a ``exists?`` check.

        Parameters
        ----------
        surface_form:
            The surface form to search for.
        source_lang:
            BCP-47 source language code.
        target_lang:
            BCP-47 target language code.

        Returns
        -------
        VocabularyEntry or None
        """
        sql = """
            SELECT * FROM vocabulary
            WHERE surface_form = ?
              AND source_lang  = ?
              AND target_lang  = ?
            LIMIT 1
        """
        try:
            row = self._conn.execute(
                sql, (surface_form, source_lang.lower(), target_lang.lower())
            ).fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to search VocabularyEntry: {exc}") from exc

        return self._row_to_vocab(row) if row else None

    def list_all(
        self,
        source_lang: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[VocabularyEntry]:
        """
        Return a paginated list of vocabulary entries, optionally filtered by
        source language.

        Parameters
        ----------
        source_lang:
            If provided, only entries matching this BCP-47 code are returned.
        limit:
            Maximum number of records to return (default 500).
        offset:
            Number of records to skip (for pagination).

        Returns
        -------
        list[VocabularyEntry]
        """
        if source_lang is not None:
            sql = """
                SELECT * FROM vocabulary
                WHERE source_lang = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            params: tuple = (source_lang.lower(), limit, offset)
        else:
            sql = """
                SELECT * FROM vocabulary
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            params = (limit, offset)

        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to list VocabularyEntries: {exc}") from exc

        return [self._row_to_vocab(row) for row in rows]

    def count(self, source_lang: Optional[str] = None) -> int:
        """
        Return the total number of stored vocabulary entries.

        Parameters
        ----------
        source_lang:
            If provided, restrict count to this language.

        Returns
        -------
        int
        """
        if source_lang is not None:
            sql = "SELECT COUNT(*) FROM vocabulary WHERE source_lang = ?"
            params_c: tuple = (source_lang.lower(),)
        else:
            sql = "SELECT COUNT(*) FROM vocabulary"
            params_c = ()

        try:
            row = self._conn.execute(sql, params_c).fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to count VocabularyEntries: {exc}") from exc

        return int(row[0])

    def get_examples(self, vocab_id: int) -> list[ExampleSentence]:
        """
        Retrieve all example sentences associated with a vocabulary entry.

        Parameters
        ----------
        vocab_id:
            The parent vocabulary entry's primary key.

        Returns
        -------
        list[ExampleSentence]
            May be empty if no examples have been added.
        """
        sql = "SELECT * FROM example_sentences WHERE vocab_id = ? ORDER BY id"
        try:
            rows = self._conn.execute(sql, (vocab_id,)).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to fetch ExampleSentences: {exc}") from exc

        return [self._row_to_example(row) for row in rows]

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, entry: VocabularyEntry) -> VocabularyEntry:
        """
        Update mutable fields of an existing :class:`VocabularyEntry`.

        Parameters
        ----------
        entry:
            An entry with a non-``None`` ``id``.  All mutable columns are
            overwritten from this object.

        Returns
        -------
        VocabularyEntry
            A new immutable instance with a refreshed ``updated_at``.

        Raises
        ------
        RecordNotFoundError
            If the ``id`` does not match any stored entry.
        DatabaseError
            For unexpected SQLite errors.
        """
        if entry.id is None:
            raise DatabaseError(
                "Cannot update a VocabularyEntry whose id is None. "
                "Use add() to persist new entries."
            )

        now = _utcnow_str()
        sql = """
            UPDATE vocabulary
            SET base_form      = ?,
                reading        = ?,
                part_of_speech = ?,
                translation    = ?,
                notes          = ?,
                review_count   = ?,
                updated_at     = ?
            WHERE id = ?
        """
        try:
            cursor = self._conn.execute(
                sql,
                (
                    entry.base_form,
                    entry.reading,
                    entry.part_of_speech,
                    entry.translation,
                    entry.notes,
                    entry.review_count,
                    now,
                    entry.id,
                ),
            )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to update VocabularyEntry: {exc}") from exc

        if cursor.rowcount == 0:
            raise RecordNotFoundError("VocabularyEntry", entry.id)

        logger.debug("Updated VocabularyEntry id=%d", entry.id)
        return entry.model_copy(update={"updated_at": _parse_dt(now)})

    def increment_review_count(self, vocab_id: int) -> VocabularyEntry:
        """
        Atomically increment the ``review_count`` of a vocabulary entry.

        Parameters
        ----------
        vocab_id:
            Primary key of the entry to update.

        Returns
        -------
        VocabularyEntry
            The updated entry.

        Raises
        ------
        RecordNotFoundError
            If the ``vocab_id`` does not exist.
        """
        now = _utcnow_str()
        sql = """
            UPDATE vocabulary
            SET review_count = review_count + 1,
                updated_at   = ?
            WHERE id = ?
        """
        try:
            cursor = self._conn.execute(sql, (now, vocab_id))
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"Failed to increment review_count for id={vocab_id}: {exc}"
            ) from exc

        if cursor.rowcount == 0:
            raise RecordNotFoundError("VocabularyEntry", vocab_id)

        return self.get_by_id(vocab_id)

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete(self, vocab_id: int) -> None:
        """
        Remove a vocabulary entry (and all cascading example sentences) by id.

        Parameters
        ----------
        vocab_id:
            Primary key to delete.

        Raises
        ------
        RecordNotFoundError
            If no row with that ``id`` exists.
        DatabaseError
            For unexpected SQLite errors.
        """
        sql = "DELETE FROM vocabulary WHERE id = ?"
        try:
            cursor = self._conn.execute(sql, (vocab_id,))
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to delete VocabularyEntry: {exc}") from exc

        if cursor.rowcount == 0:
            raise RecordNotFoundError("VocabularyEntry", vocab_id)

        logger.debug("Deleted VocabularyEntry id=%d (cascade applied)", vocab_id)

    def delete_all(self) -> int:
        """
        Remove all vocabulary entries and all cascading example sentences.

        Returns
        -------
        int
            Number of vocabulary entries deleted.

        Raises
        ------
        DatabaseError
            For unexpected SQLite errors.
        """
        try:
            with self._conn:
                self._conn.execute("DELETE FROM example_sentences")
                cursor = self._conn.execute("DELETE FROM vocabulary")
                count = cursor.rowcount
            logger.debug("Deleted all %d VocabularyEntry records", count)
            return count
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to delete all VocabularyEntry records: {exc}") from exc

