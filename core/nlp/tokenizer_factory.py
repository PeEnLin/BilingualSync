"""
core/nlp/tokenizer_factory.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Factory that maps BCP-47 language codes to concrete tokeniser instances.

Usage
-----
>>> from core.nlp.tokenizer_factory import TokenizerFactory
>>> tok = TokenizerFactory.get("ja")
>>> tok.tokenize("走った")

Design
------
* :meth:`TokenizerFactory.get` is the single call-site entry point.
* Engines are **lazily instantiated** and cached per-process (module-level
  ``_REGISTRY`` dict) so heavyweight Janome initialisation happens only once.
* New languages are registered via :meth:`TokenizerFactory.register`, which
  lets tests and plugins inject mock tokenisers without subclassing.
* Unknown language codes raise :class:`~core.exceptions.UnsupportedLanguageError`.
"""

from __future__ import annotations

import logging
from typing import Callable

from core.base import BaseTokenizer
from core.exceptions import UnsupportedLanguageError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal registry
# ─────────────────────────────────────────────────────────────────────────────

# Maps normalised language code → factory callable that returns a BaseTokenizer.
# Factories are zero-argument callables so that instantiation is deferred.
_FACTORIES: dict[str, Callable[[], BaseTokenizer]] = {}

# Cache of already-created instances keyed by normalised lang code.
_INSTANCES: dict[str, BaseTokenizer] = {}


def _normalise(lang_code: str) -> str:
    """Lower-case and strip a language code (e.g. ``"JA"`` → ``"ja"``)."""
    return lang_code.strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
# Public factory class
# ─────────────────────────────────────────────────────────────────────────────

class TokenizerFactory:
    """
    Static factory for language-specific tokeniser instances.

    All public methods are class-methods so the factory is used without
    instantiation.
    """

    @classmethod
    def register(
        cls,
        lang_code: str,
        factory: Callable[[], BaseTokenizer],
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register a factory callable for a language code.

        Parameters
        ----------
        lang_code:
            BCP-47 language code (case-insensitive, e.g. ``"ja"``, ``"zh-TW"``).
        factory:
            Zero-argument callable that returns a :class:`~core.base.BaseTokenizer`
            instance when called.
        overwrite:
            If ``False`` (default) and the language is already registered, a
            ``ValueError`` is raised.  Set to ``True`` to replace the existing
            factory (useful in tests).

        Raises
        ------
        ValueError
            If ``overwrite=False`` and the language is already registered.
        """
        key = _normalise(lang_code)
        if key in _FACTORIES and not overwrite:
            raise ValueError(
                f"Language '{key}' is already registered. "
                "Pass overwrite=True to replace the existing factory."
            )
        _FACTORIES[key] = factory
        # Invalidate any cached instance so the new factory is used next time.
        _INSTANCES.pop(key, None)
        logger.debug("Registered tokeniser factory for lang_code=%r.", key)

    @classmethod
    def get(cls, lang_code: str) -> BaseTokenizer:
        """
        Return a cached tokeniser for ``lang_code``, creating it on first call.

        Parameters
        ----------
        lang_code:
            BCP-47 language code (e.g. ``"ja"``, ``"zh-TW"``).

        Returns
        -------
        BaseTokenizer
            Concrete tokeniser implementation for the requested language.

        Raises
        ------
        UnsupportedLanguageError
            If no factory is registered for ``lang_code``.
        EngineInitialisationError
            If the factory callable raises during engine construction.
        """
        key = _normalise(lang_code)

        if key in _INSTANCES:
            return _INSTANCES[key]

        if key not in _FACTORIES:
            raise UnsupportedLanguageError(lang_code=lang_code)

        logger.debug("Instantiating tokeniser for lang_code=%r.", key)
        instance = _FACTORIES[key]()
        _INSTANCES[key] = instance
        return instance

    @classmethod
    def supported_languages(cls) -> list[str]:
        """Return a sorted list of registered language codes."""
        return sorted(_FACTORIES.keys())

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear the instance cache.

        Useful in tests to force re-initialisation between test cases without
        deregistering factories.
        """
        _INSTANCES.clear()

    @classmethod
    def reset(cls) -> None:
        """
        Clear both the factory registry and the instance cache.

        Use in tests that need a completely clean factory state.
        """
        _FACTORIES.clear()
        _INSTANCES.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Built-in registrations
# ─────────────────────────────────────────────────────────────────────────────

def _register_defaults() -> None:
    """Register all built-in language engines."""
    from core.nlp.japanese_engine import JapaneseTokenizer  # noqa: PLC0415

    TokenizerFactory.register("ja", JapaneseTokenizer)


_register_defaults()
