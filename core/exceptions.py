"""
core/exceptions.py
~~~~~~~~~~~~~~~~~~
Unified exception hierarchy for the BilingualSync application.

All domain-specific exceptions inherit from :class:`BilingualSyncError` so
callers can catch the entire family with a single ``except BilingualSyncError``
clause while still distinguishing sub-types when finer handling is required.

Design notes
------------
* Never raise bare ``Exception`` or ``BaseException`` inside this codebase.
* Every exception carries a human-readable ``message`` and an optional
  ``context`` dict for structured logging / debugging.
* Use the most specific subclass available; add new ones here rather than
  in individual modules.
"""

from __future__ import annotations

from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Root Exception
# ─────────────────────────────────────────────────────────────────────────────

class BilingualSyncError(Exception):
    """
    Root exception for all BilingualSync domain errors.

    Parameters
    ----------
    message:
        Human-readable error description.
    context:
        Optional key-value pairs with structured debugging information
        (e.g., file paths, offending input, line numbers).
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.context: dict[str, Any] = context or {}

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(message={self.message!r}, context={self.context!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Errors
# ─────────────────────────────────────────────────────────────────────────────

class ConfigurationError(BilingualSyncError):
    """Raised when the application configuration is invalid or incomplete."""


# ─────────────────────────────────────────────────────────────────────────────
# SRT / Subtitle Parsing Errors
# ─────────────────────────────────────────────────────────────────────────────

class SRTParseError(BilingualSyncError):
    """Base class for SRT subtitle parsing failures."""


class CorruptedSRTError(SRTParseError):
    """
    Raised when an SRT file is structurally corrupted beyond auto-repair.

    Parameters
    ----------
    message:
        Description of what is corrupted.
    file_path:
        Path to the offending SRT file (injected into ``context``).
    line_number:
        Approximate line number where corruption was detected.
    """

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        line_number: int | None = None,
    ) -> None:
        context: dict[str, Any] = {}
        if file_path is not None:
            context["file_path"] = file_path
        if line_number is not None:
            context["line_number"] = line_number
        super().__init__(message, context)


class MalformedTimestampError(SRTParseError):
    """Raised when an SRT timestamp cannot be parsed into a valid time value."""

    def __init__(self, raw_timestamp: str, line_number: int | None = None) -> None:
        context: dict[str, Any] = {"raw_timestamp": raw_timestamp}
        if line_number is not None:
            context["line_number"] = line_number
        super().__init__(
            f"Cannot parse SRT timestamp: {raw_timestamp!r}",
            context,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Alignment Errors
# ─────────────────────────────────────────────────────────────────────────────

class AlignmentError(BilingualSyncError):
    """Raised when bilingual sentence alignment fails irrecoverably."""


class IncompatibleSubtitleLengthError(AlignmentError):
    """
    Raised when source and target subtitle tracks have incompatible lengths
    and automatic alignment cannot produce a reliable mapping.
    """

    def __init__(self, source_count: int, target_count: int) -> None:
        super().__init__(
            f"Cannot align subtitles: source has {source_count} blocks, "
            f"target has {target_count} blocks.",
            {"source_count": source_count, "target_count": target_count},
        )


# ─────────────────────────────────────────────────────────────────────────────
# NLP / Tokenisation Errors
# ─────────────────────────────────────────────────────────────────────────────

class TokenizationError(BilingualSyncError):
    """Base class for tokenisation failures."""


class UnsupportedLanguageError(TokenizationError):
    """
    Raised when a requested language code has no registered tokeniser engine.

    Parameters
    ----------
    lang_code:
        The BCP-47 language code that was requested but is unsupported.
    """

    def __init__(self, lang_code: str) -> None:
        super().__init__(
            f"No tokeniser engine registered for language code: {lang_code!r}",
            {"lang_code": lang_code},
        )


class EngineInitialisationError(TokenizationError):
    """Raised when a tokeniser engine fails to initialise (e.g., missing model)."""


# ─────────────────────────────────────────────────────────────────────────────
# Database / Repository Errors
# ─────────────────────────────────────────────────────────────────────────────

class DatabaseError(BilingualSyncError):
    """Base class for all database interaction failures."""


class RecordNotFoundError(DatabaseError):
    """
    Raised when a requested record does not exist in the repository.

    Parameters
    ----------
    entity:
        Name of the domain entity (e.g., ``"VocabularyEntry"``).
    identifier:
        The ID or key used to look up the missing record.
    """

    def __init__(self, entity: str, identifier: Any) -> None:
        super().__init__(
            f"{entity} with identifier {identifier!r} was not found.",
            {"entity": entity, "identifier": identifier},
        )


class DuplicateRecordError(DatabaseError):
    """
    Raised when an insert would violate a uniqueness constraint.

    Parameters
    ----------
    entity:
        Name of the domain entity.
    duplicate_field:
        Field name (or composite key) that caused the collision.
    value:
        The duplicate value.
    """

    def __init__(self, entity: str, duplicate_field: str, value: Any) -> None:
        super().__init__(
            f"Duplicate {entity}: a record with {duplicate_field}={value!r} already exists.",
            {"entity": entity, "field": duplicate_field, "value": value},
        )


class SchemaError(DatabaseError):
    """Raised when the database schema is missing or incompatible."""


# ─────────────────────────────────────────────────────────────────────────────
# Export Errors
# ─────────────────────────────────────────────────────────────────────────────

class ExportError(BilingualSyncError):
    """Base class for export-related failures."""


class UnsupportedExportFormatError(ExportError):
    """Raised when an unknown export format is requested."""

    def __init__(self, fmt: str) -> None:
        super().__init__(
            f"Export format {fmt!r} is not supported.",
            {"requested_format": fmt},
        )
