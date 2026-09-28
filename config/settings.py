"""
config/settings.py
~~~~~~~~~~~~~~~~~~
Application-wide configuration management using Pydantic Settings.

Supports loading from environment variables (prefix: BILINGUALSYNC_) and
.env files. All fields are strongly typed with production-safe defaults.

Usage::

    from config.settings import get_settings
    settings = get_settings()
    print(settings.db_path)
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralised, immutable application settings.

    All attributes can be overridden via environment variables using the
    prefix ``BILINGUALSYNC_``, e.g.::

        export BILINGUALSYNC_DB_PATH=/data/bilingual.db

    Attributes
    ----------
    db_path:
        Absolute or relative path to the SQLite database file.
    supported_languages:
        Tuple of BCP-47 language codes the system can tokenise.
    default_source_lang:
        The source (foreign) language code used when none is specified.
    default_target_lang:
        The target (native) language code used when none is specified.
    tokenizer_cache_size:
        LRU cache size (number of entries) for tokenisation results.
    export_dir:
        Directory where exported Anki / JSON files are written.
    log_level:
        Python logging level string (DEBUG / INFO / WARNING / ERROR).
    """

    model_config = SettingsConfigDict(
        env_prefix="BILINGUALSYNC_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        frozen=True,                  # Enforce immutability at runtime
    )

    # ── Database ─────────────────────────────────────────────────────────────
    db_path: Path = Field(
        default=Path("bilingual_sync.db"),
        description="Path to the SQLite database file.",
    )

    # ── Language Settings ─────────────────────────────────────────────────────
    supported_languages: tuple[str, ...] = Field(
        default=("ja", "zh-TW", "en", "ko"),
        description="BCP-47 codes for supported languages.",
    )
    default_source_lang: str = Field(
        default="ja",
        description="Default source (foreign) language code.",
    )
    default_target_lang: str = Field(
        default="zh-TW",
        description="Default target (native) language code.",
    )

    # ── NLP / Cache ───────────────────────────────────────────────────────────
    tokenizer_cache_size: int = Field(
        default=512,
        ge=64,
        le=8192,
        description="Maximum number of tokenisation results to cache (LRU).",
    )

    # ── Export ────────────────────────────────────────────────────────────────
    export_dir: Path = Field(
        default=Path("exports"),
        description="Directory where exported files are written.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Python logging level.",
    )

    # ── Aligner ───────────────────────────────────────────────────────────────
    aligner_window_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Sliding-window size (in subtitle blocks) for bilingual alignment.",
    )

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("default_source_lang", "default_target_lang", mode="before")
    @classmethod
    def _validate_lang_code(cls, v: str) -> str:
        """Normalise language code to lower-case BCP-47 format."""
        return v.strip().lower()

    @field_validator("db_path", "export_dir", mode="before")
    @classmethod
    def _coerce_to_path(cls, v: object) -> Path:
        """Accept strings and coerce them to ``pathlib.Path``."""
        return Path(str(v))


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a module-level singleton :class:`Settings` instance.

    The result is cached via ``functools.lru_cache`` so that environment
    variables are read exactly once per process.

    Returns
    -------
    Settings
        The application-wide settings singleton.
    """
    return Settings()
