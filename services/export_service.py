"""
services/export_service.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Export service for the BilingualSync vocabulary store.

:class:`ExportService` converts persisted :class:`~database.models.VocabularyEntry`
records (and their :class:`~database.models.ExampleSentence` siblings) into
portable file formats suitable for spaced-repetition or backup workflows.

Supported formats
-----------------
``export_to_anki``
    Tab-separated values (TSV) or custom-delimiter CSV that can be imported
    directly into Anki's *Import* dialog.  Columns::

        surface | base_form | reading | part_of_speech |
        source_example | target_example | created_at

    All text is HTML-escaped so angle-bracket characters in subtitle text
    do not break Anki's HTML renderer.

``export_to_json``
    Structured JSON backup containing every field of every entry together
    with the full list of associated example sentences.

``export_to_csv``
    Standard RFC-4180 CSV with a configurable delimiter (defaults to comma).
    Useful for spreadsheet / LibreOffice workflows.

All writes use UTF-8 encoding with a BOM (``utf-8-sig``) for maximum
compatibility with Windows applications including Microsoft Excel and Anki.
"""

from __future__ import annotations

import csv
import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


from database.models import ExampleSentence, VocabularyEntry
from database.repository import VocabularyRepository
from core.exceptions import ExportError, UnsupportedExportFormatError

logger = logging.getLogger(__name__)

# Anki TSV column order
_ANKI_FIELDNAMES = [
    "surface",
    "base_form",
    "reading",
    "part_of_speech",
    "source_example",
    "target_example",
    "created_at",
]


def _dt_str(dt: datetime) -> str:
    """Format a datetime as a compact ISO-8601 string (second resolution)."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _first_example(
    examples: list[ExampleSentence],
) -> tuple[str, str]:
    """Return ``(source_text, target_text)`` of the first example, or ``("", "")``."""
    if examples:
        ex = examples[0]
        return ex.source_text, ex.target_text
    return "", ""


class ExportService:
    """
    Export vocabulary entries to file.

    Parameters
    ----------
    repo:
        An open :class:`~database.repository.VocabularyRepository`.
    source_lang:
        BCP-47 source language filter (default ``"ja"``).  Only entries for
        this language are included in exports.
    """

    def __init__(
        self,
        repo: VocabularyRepository,
        source_lang: str = "ja",
    ) -> None:
        self._repo = repo
        self.source_lang = source_lang.strip().lower()

    # ─────────────────────────────────────────────────────────────────────────
    # Public export methods
    # ─────────────────────────────────────────────────────────────────────────

    def export_to_anki(
        self,
        file_path: str | Path,
        delimiter: str = "\t",
        include_header: bool = False,
    ) -> int:
        """
        Export vocabulary entries to an Anki-compatible TSV/CSV file.

        Anki import format (one note per line)::

            surface<TAB>base_form<TAB>reading<TAB>part_of_speech<TAB>
            source_example<TAB>target_example<TAB>created_at

        All cell values are HTML-escaped so subtitle text with ``<`` / ``>``
        characters does not corrupt Anki's HTML renderer.

        Parameters
        ----------
        file_path:
            Destination file path.  Directories are created if absent.
        delimiter:
            Field delimiter (default ``"\\t"`` for TSV; use ``","`` for CSV).
        include_header:
            If ``True``, a ``#`` comment header line listing field names is
            prepended (Anki ignores lines starting with ``#``).

        Returns
        -------
        int
            Number of rows written.

        Raises
        ------
        ExportError
            If the file cannot be written.
        """
        entries = self._load_entries()
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows_written = 0
        try:
            with path.open("w", encoding="utf-8-sig", newline="") as fh:
                if include_header:
                    fh.write("#" + delimiter.join(_ANKI_FIELDNAMES) + "\n")

                writer = csv.writer(fh, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
                for entry in entries:
                    examples = self._repo.get_examples(entry.id)  # type: ignore[arg-type]
                    src_ex, tgt_ex = _first_example(examples)
                    row = [
                        html.escape(entry.surface_form),
                        html.escape(entry.base_form),
                        html.escape(entry.reading),
                        html.escape(entry.part_of_speech),
                        html.escape(src_ex),
                        html.escape(tgt_ex),
                        _dt_str(entry.created_at),
                    ]
                    writer.writerow(row)
                    rows_written += 1
        except OSError as exc:
            raise ExportError(
                f"Failed to write Anki export to {path}: {exc}",
                {"file_path": str(path), "original_error": str(exc)},
            ) from exc

        logger.info("Anki export: %d rows → %s", rows_written, path)
        return rows_written

    def export_to_json(
        self,
        file_path: str | Path,
        indent: int = 2,
    ) -> int:
        """
        Export all vocabulary entries and their example sentences to JSON.

        Structure::

            {
              "exported_at": "ISO-8601 timestamp",
              "source_lang": "ja",
              "count": 3,
              "entries": [
                {
                  "id": 1,
                  "surface_form": "食べた",
                  "base_form": "食べる",
                  ...
                  "examples": [
                    {"source_text": "...", "target_text": "...", ...}
                  ]
                },
                ...
              ]
            }

        Parameters
        ----------
        file_path:
            Destination file path.  Parent directories are created as needed.
        indent:
            JSON indentation level (default 2).

        Returns
        -------
        int
            Number of vocabulary entries serialised.

        Raises
        ------
        ExportError
            If the file cannot be written.
        """
        entries = self._load_entries()
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict = {
            "exported_at": _dt_str(datetime.now(timezone.utc)),
            "source_lang": self.source_lang,
            "count": len(entries),
            "entries": [],
        }

        for entry in entries:
            examples = self._repo.get_examples(entry.id)  # type: ignore[arg-type]
            entry_dict = {
                "id": entry.id,
                "surface_form": entry.surface_form,
                "base_form": entry.base_form,
                "reading": entry.reading,
                "part_of_speech": entry.part_of_speech,
                "source_lang": entry.source_lang,
                "target_lang": entry.target_lang,
                "translation": entry.translation,
                "notes": entry.notes,
                "review_count": entry.review_count,
                "created_at": _dt_str(entry.created_at),
                "updated_at": _dt_str(entry.updated_at),
                "examples": [
                    {
                        "id": ex.id,
                        "source_text": ex.source_text,
                        "target_text": ex.target_text,
                        "source_start_ms": ex.source_start_ms,
                        "source_end_ms": ex.source_end_ms,
                        "created_at": _dt_str(ex.created_at),
                    }
                    for ex in examples
                ],
            }
            payload["entries"].append(entry_dict)

        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=indent)
        except OSError as exc:
            raise ExportError(
                f"Failed to write JSON export to {path}: {exc}",
                {"file_path": str(path), "original_error": str(exc)},
            ) from exc

        logger.info("JSON export: %d entries → %s", len(entries), path)
        return len(entries)

    def export_to_csv(
        self,
        file_path: str | Path,
        delimiter: str = ",",
    ) -> int:
        """
        Export vocabulary entries to a standard RFC-4180 CSV file.

        Equivalent to :meth:`export_to_anki` with a comma delimiter and a
        visible header row (not Anki-comment style).

        Parameters
        ----------
        file_path:
            Destination file path.
        delimiter:
            Field delimiter (default comma).

        Returns
        -------
        int
            Number of data rows written (excluding header).

        Raises
        ------
        ExportError
            If the file cannot be written.
        """
        entries = self._load_entries()
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows_written = 0
        try:
            with path.open("w", encoding="utf-8-sig", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=_ANKI_FIELDNAMES,
                    delimiter=delimiter,
                    quoting=csv.QUOTE_ALL,
                )
                writer.writeheader()
                for entry in entries:
                    examples = self._repo.get_examples(entry.id)  # type: ignore[arg-type]
                    src_ex, tgt_ex = _first_example(examples)
                    writer.writerow({
                        "surface": entry.surface_form,
                        "base_form": entry.base_form,
                        "reading": entry.reading,
                        "part_of_speech": entry.part_of_speech,
                        "source_example": src_ex,
                        "target_example": tgt_ex,
                        "created_at": _dt_str(entry.created_at),
                    })
                    rows_written += 1
        except OSError as exc:
            raise ExportError(
                f"Failed to write CSV export to {path}: {exc}",
                {"file_path": str(path), "original_error": str(exc)},
            ) from exc

        logger.info("CSV export: %d rows → %s", rows_written, path)
        return rows_written

    def export(
        self,
        fmt: str,
        file_path: str | Path,
        **kwargs,
    ) -> int:
        """
        Dispatch export by format string.

        Parameters
        ----------
        fmt:
            One of ``"anki"``, ``"json"``, ``"csv"`` (case-insensitive).
        file_path:
            Destination path.
        **kwargs:
            Passed through to the underlying export method.

        Raises
        ------
        UnsupportedExportFormatError
            If ``fmt`` is not recognised.
        """
        dispatch: dict[str, Callable] = {
            "anki": self.export_to_anki,
            "json": self.export_to_json,
            "csv": self.export_to_csv,
        }
        key = fmt.strip().lower()
        if key not in dispatch:
            raise UnsupportedExportFormatError(fmt)
        return dispatch[key](file_path, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load_entries(self) -> list[VocabularyEntry]:
        """Load all entries for ``source_lang`` (up to 100 000 rows)."""
        return self._repo.list_all(source_lang=self.source_lang, limit=100_000)
