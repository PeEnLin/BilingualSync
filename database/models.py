"""
database/models.py
~~~~~~~~~~~~~~~~~~
Domain entities (Pydantic models) representing the core data objects stored
in the BilingualSync SQLite database.

Design decisions
----------------
* All models inherit from ``pydantic.BaseModel`` for validation and
  serialisation (JSON / dict) out-of-the-box.
* ``id`` fields are ``Optional[int]`` (``None`` before first insert) to
  distinguish transient from persisted entities.
* ``created_at`` / ``updated_at`` are ``datetime`` objects managed by the
  repository layer (not the database trigger) for portability across SQLite
  versions.
* ``model_config = ConfigDict(frozen=True)`` makes persisted entities
  effectively immutable. Use ``model_copy(update={...})`` to produce a
  mutated copy when updating.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utcnow() -> datetime:
    """Return the current UTC timestamp (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# VocabularyEntry
# ─────────────────────────────────────────────────────────────────────────────

class VocabularyEntry(BaseModel):
    """
    A single vocabulary item harvested from a bilingual subtitle corpus.

    Attributes
    ----------
    id:
        Auto-assigned primary key (``None`` for unsaved entities).
    surface_form:
        The word exactly as it appears in the source text (e.g. ``"食べた"``).
    base_form:
        Dictionary / lemma form produced by the NLP engine (e.g. ``"食べる"``).
    reading:
        Phonetic reading, e.g. hiragana for Japanese (``"たべた"``).
    part_of_speech:
        Part-of-speech tag returned by the tokeniser (e.g. ``"動詞"``).
    source_lang:
        BCP-47 code of the source language (e.g. ``"ja"``).
    target_lang:
        BCP-47 code of the target language (e.g. ``"zh-TW"``).
    translation:
        Human-verified translation of the word in the target language.
    notes:
        Free-form learner notes.
    review_count:
        Number of times this entry has been reviewed (SRS support).
    created_at:
        UTC timestamp when the entry was first inserted.
    updated_at:
        UTC timestamp of the most recent update.
    """

    model_config = ConfigDict(frozen=True)

    id: Optional[int] = Field(default=None, description="Primary key (None = unsaved).")
    surface_form: str = Field(
        min_length=1,
        description="Word as it appears in source text.",
    )
    base_form: str = Field(
        min_length=1,
        description="Dictionary / lemma form.",
    )
    reading: str = Field(
        default="",
        description="Phonetic reading (hiragana / romanji / pinyin etc.).",
    )
    part_of_speech: str = Field(
        default="",
        description="Part-of-speech tag from the NLP engine.",
    )
    source_lang: str = Field(
        default="ja",
        description="BCP-47 source language code.",
    )
    target_lang: str = Field(
        default="zh-TW",
        description="BCP-47 target language code.",
    )
    translation: str = Field(
        default="",
        description="Target-language translation of this vocabulary item.",
    )
    notes: str = Field(
        default="",
        description="Learner notes / mnemonics.",
    )
    review_count: int = Field(
        default=0,
        ge=0,
        description="Spaced-repetition review counter.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of initial insertion.",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of last modification.",
    )

    @field_validator("surface_form", "base_form", mode="before")
    @classmethod
    def _strip_whitespace(cls, v: str) -> str:
        """Trim leading/trailing whitespace from text fields."""
        return v.strip()

    @field_validator("source_lang", "target_lang", mode="before")
    @classmethod
    def _normalise_lang(cls, v: str) -> str:
        """Normalise language codes to lower-case."""
        return v.strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
# ExampleSentence
# ─────────────────────────────────────────────────────────────────────────────

class ExampleSentence(BaseModel):
    """
    A bilingual example sentence associated with a :class:`VocabularyEntry`.

    Attributes
    ----------
    id:
        Auto-assigned primary key.
    vocab_id:
        Foreign key referencing ``VocabularyEntry.id``.
    source_text:
        Original source-language sentence (e.g. Japanese subtitle text).
    target_text:
        Corresponding target-language sentence (e.g. Chinese translation).
    source_start_ms:
        Optional subtitle start timestamp in milliseconds.
    source_end_ms:
        Optional subtitle end timestamp in milliseconds.
    created_at:
        UTC timestamp when the sentence was first inserted.
    """

    model_config = ConfigDict(frozen=True)

    id: Optional[int] = Field(default=None, description="Primary key (None = unsaved).")
    vocab_id: int = Field(description="Foreign key to VocabularyEntry.")
    source_text: str = Field(min_length=1, description="Source-language sentence.")
    target_text: str = Field(default="", description="Target-language translation.")
    source_start_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Subtitle start time in milliseconds.",
    )
    source_end_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Subtitle end time in milliseconds.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of insertion.",
    )
