"""
services/vocab_service.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Application-layer service that orchestrates vocabulary extraction, NLP
analysis, persistence, and retrieval.

:class:`VocabularyService` is the primary entry point for UI interactions such
as "user clicked a word in the subtitle view".  It bridges:

* The **NLP layer** (:class:`~core.base.BaseTokenizer`) – tokenises raw text
  and extracts morphological features.
* The **repository layer** (:class:`~database.repository.VocabularyRepository`)
  – persists and retrieves vocabulary entries.

Design decisions
----------------
* The service is **stateless** with respect to database transactions; it
  delegates connection/transaction management to callers via the repository.
* ``extract_and_save_word`` implements an **upsert-like** strategy:
  - If the surface form already exists for the same language pair the existing
    entry is returned unchanged (no duplicate, no error).
  - If it does not exist the NLP engine is consulted and the result is saved.
* ``search`` performs a case-insensitive LIKE query via
  :meth:`~database.repository.VocabularyRepository.list_all` filtered in
  Python, keeping SQL simple and avoiding custom SQL in this layer.
"""

from __future__ import annotations

import logging
from typing import Optional

from core.base import BaseTokenizer, Token
from core.exceptions import TokenizationError
from database.models import ExampleSentence, VocabularyEntry
from database.repository import VocabularyRepository

logger = logging.getLogger(__name__)


class VocabularyService:
    """
    Application-layer façade for vocabulary management.

    Parameters
    ----------
    repo:
        An open :class:`~database.repository.VocabularyRepository` instance.
    tokenizer:
        A concrete :class:`~core.base.BaseTokenizer` (e.g.
        :class:`~core.nlp.japanese_engine.JapaneseTokenizer`).
    source_lang:
        BCP-47 code of the source language being processed (default ``"ja"``).
    target_lang:
        BCP-47 code of the target language (default ``"zh-tw"``).
    """

    def __init__(
        self,
        repo: VocabularyRepository,
        tokenizer: BaseTokenizer,
        source_lang: str = "ja",
        target_lang: str = "zh-tw",
    ) -> None:
        self._repo = repo
        self._tokenizer = tokenizer
        self.source_lang = source_lang.strip().lower()
        self.target_lang = target_lang.strip().lower()

    # ─────────────────────────────────────────────────────────────────────────
    # Core: extract & save
    # ─────────────────────────────────────────────────────────────────────────

    def extract_and_save_word(
        self,
        clicked_text: str,
        source_sentence: str = "",
        target_sentence: str = "",
        translation: str = "",
        notes: str = "",
        source_start_ms: Optional[int] = None,
        source_end_ms: Optional[int] = None,
    ) -> VocabularyEntry:
        """
        Tokenise ``clicked_text``, look up morphological features, and persist
        the result as a :class:`~database.models.VocabularyEntry`.

        If an entry with the same ``(surface_form, source_lang, target_lang)``
        already exists, the existing record is returned immediately (idempotent
        upsert semantics) and the example sentence – if provided – is still
        appended.

        Parameters
        ----------
        clicked_text:
            The word or phrase the user clicked / selected in the UI.
            Leading/trailing whitespace is stripped.
        source_sentence:
            The complete source-language sentence containing ``clicked_text``.
            Used to create an :class:`~database.models.ExampleSentence` when
            non-empty.
        target_sentence:
            The corresponding translated sentence (stored as example context).
        translation:
            Optional human-provided translation override for the word itself.
        notes:
            Free-form learner notes.
        source_start_ms:
            Subtitle start timestamp (ms) for the example sentence.
        source_end_ms:
            Subtitle end timestamp (ms) for the example sentence.

        Returns
        -------
        VocabularyEntry
            The persisted vocabulary entry (new or pre-existing).

        Raises
        ------
        ValueError
            If ``clicked_text`` is blank after stripping.
        TokenizationError
            If the NLP engine fails to analyse the text.
        """
        surface = clicked_text.strip()
        if not surface:
            raise ValueError("clicked_text must not be empty.")

        # ── 1. Idempotency check ──────────────────────────────────────────
        existing = self._repo.find_by_surface(
            surface, source_lang=self.source_lang, target_lang=self.target_lang
        )
        if existing is not None:
            logger.debug("Surface %r already saved (id=%d); skipping NLP.", surface, existing.id)
            entry = existing
        else:
            # ── 2. NLP analysis ───────────────────────────────────────────
            token = self._find_token(surface)

            entry = VocabularyEntry(
                surface_form=surface,
                base_form=token.base_form if token else surface,
                reading=token.reading if token else "",
                part_of_speech=token.part_of_speech if token else "",
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                translation=translation,
                notes=notes,
            )
            entry = self._repo.add(entry)
            logger.info("Saved new VocabularyEntry id=%d surface=%r", entry.id, surface)

        # ── 3. Attach example sentence (always, even for existing entries) ─
        if source_sentence.strip():
            try:
                self._repo.add_example(
                    ExampleSentence(
                        vocab_id=entry.id,  # type: ignore[arg-type]
                        source_text=source_sentence.strip(),
                        target_text=target_sentence.strip(),
                        source_start_ms=source_start_ms,
                        source_end_ms=source_end_ms,
                    )
                )
                logger.debug(
                    "Attached example sentence to VocabularyEntry id=%d.", entry.id
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to attach example sentence: %s", exc)

        return entry

    # ─────────────────────────────────────────────────────────────────────────
    # Read operations
    # ─────────────────────────────────────────────────────────────────────────

    def get_all_saved_words(
        self,
        limit: int = 500,
        offset: int = 0,
    ) -> list[VocabularyEntry]:
        """
        Return all stored vocabulary entries for the configured language pair.

        Parameters
        ----------
        limit:
            Maximum number of records (default 500).
        offset:
            Pagination offset.

        Returns
        -------
        list[VocabularyEntry]
            Ordered newest-first.
        """
        return self._repo.list_all(
            source_lang=self.source_lang,
            limit=limit,
            offset=offset,
        )

    def get_word_count(self) -> int:
        """Return the count of saved words for the configured source language."""
        return self._repo.count(source_lang=self.source_lang)

    def get_examples(self, vocab_id: int) -> list[ExampleSentence]:
        """
        Return all example sentences for a vocabulary entry.

        Parameters
        ----------
        vocab_id:
            Primary key of the :class:`~database.models.VocabularyEntry`.

        Returns
        -------
        list[ExampleSentence]
        """
        return self._repo.get_examples(vocab_id)

    def search(self, query: str) -> list[VocabularyEntry]:
        """
        Search saved words by surface form, base form, or translation.

        Performs a case-insensitive substring match.  Results are returned for
        the configured ``source_lang`` only.

        Parameters
        ----------
        query:
            Substring to search for (stripped; empty returns all words).

        Returns
        -------
        list[VocabularyEntry]
        """
        q = query.strip().lower()
        all_entries = self._repo.list_all(source_lang=self.source_lang, limit=10_000)
        if not q:
            return all_entries
        return [
            e for e in all_entries
            if q in e.surface_form.lower()
            or q in e.base_form.lower()
            or q in e.translation.lower()
            or q in e.reading.lower()
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Write operations
    # ─────────────────────────────────────────────────────────────────────────

    def update_translation(self, vocab_id: int, translation: str) -> VocabularyEntry:
        """
        Set the human-verified translation for an existing vocabulary entry.

        Parameters
        ----------
        vocab_id:
            Primary key of the entry to update.
        translation:
            New translation string.

        Returns
        -------
        VocabularyEntry
            The updated entry.
        """
        entry = self._repo.get_by_id(vocab_id)
        return self._repo.update(entry.model_copy(update={"translation": translation}))

    def update_notes(self, vocab_id: int, notes: str) -> VocabularyEntry:
        """
        Set the learner notes for an existing vocabulary entry.

        Parameters
        ----------
        vocab_id:
            Primary key of the entry to update.
        notes:
            New notes string.

        Returns
        -------
        VocabularyEntry
            The updated entry.
        """
        entry = self._repo.get_by_id(vocab_id)
        return self._repo.update(entry.model_copy(update={"notes": notes}))

    def delete_word(self, vocab_id: int) -> None:
        """
        Delete a vocabulary entry and all its example sentences.

        Parameters
        ----------
        vocab_id:
            Primary key of the entry to delete.

        Raises
        ------
        RecordNotFoundError
            If the entry does not exist.
        """
        self._repo.delete(vocab_id)
        logger.info("Deleted VocabularyEntry id=%d.", vocab_id)

    def delete_all_words(self) -> int:
        """
        Delete all vocabulary entries and their associated example sentences.

        Returns
        -------
        int
            The number of entries deleted.
        """
        count = self._repo.delete_all()
        logger.info("Deleted all vocabulary entries (count=%d).", count)
        return count

    def mark_reviewed(self, vocab_id: int) -> VocabularyEntry:
        """
        Increment the SRS review counter for a vocabulary entry.

        Parameters
        ----------
        vocab_id:
            Primary key of the entry to mark.

        Returns
        -------
        VocabularyEntry
            The updated entry with incremented ``review_count``.
        """
        return self._repo.increment_review_count(vocab_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _find_token(self, surface: str) -> Optional[Token]:
        """
        Tokenise ``surface`` and return the :class:`~core.base.Token` whose
        ``surface`` field best matches ``surface`` (exact match first, then
        the first content token).

        Returns ``None`` if tokenisation yields no tokens.
        """
        try:
            tokens = self._tokenizer.tokenize(surface)
        except TokenizationError as exc:
            logger.warning("NLP tokenisation failed for %r: %s. Saving with surface only.", surface, exc)
            return None

        if not tokens:
            return None

        # Prefer an exact surface match
        for tok in tokens:
            if tok.surface == surface:
                return tok

        # Fall back to the first token (covers split-morpheme cases)
        return tokens[0]
