"""
core/base.py
~~~~~~~~~~~~
Abstract base classes (ABCs) that define the interface contracts for the
BilingualSync core engine.

All concrete implementations of aligners and tokenisers **must** inherit from
the relevant ABC and implement every abstract method.  This guarantees that
higher-level orchestration code can swap implementations (e.g., swap Janome
for a different NLP engine) without touching call sites.

Classes
-------
BaseAligner
    Contract for bilingual subtitle alignment engines.
BaseTokenizer
    Contract for language-specific tokenisation engines.
Token
    Immutable dataclass representing a single analysed token.
AlignedPair
    Immutable dataclass representing one matched source↔target subtitle pair.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Sequence


# ─────────────────────────────────────────────────────────────────────────────
# Data Transfer Objects
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SubtitleBlock:
    """
    A single parsed subtitle block from an SRT file.

    Parameters
    ----------
    index:
        Sequential block number (1-based, as written in the SRT file).
    start_ms:
        Start time in milliseconds.
    end_ms:
        End time in milliseconds.
    text:
        Raw subtitle text (may contain multiple lines).
    """

    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        """Duration of the subtitle block in milliseconds."""
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class AlignedPair:
    """
    A matched source↔target subtitle pair produced by an aligner.

    Parameters
    ----------
    source:
        The source-language subtitle block.
    target:
        The target-language subtitle block, or ``None`` if no match was found.
    overlap_ratio:
        Temporal overlap ratio in ``[0.0, 1.0]`` used to score the match.
    """

    source: SubtitleBlock
    target: SubtitleBlock | None
    overlap_ratio: float = field(default=0.0)


@dataclass(frozen=True)
class Token:
    """
    A single morpheme produced by a tokenisation engine.

    Parameters
    ----------
    surface:
        The token exactly as it appears in the source text.
    base_form:
        Dictionary / lemma form (e.g. ``"食べる"`` for ``"食べた"``).
    reading:
        Phonetic reading (hiragana, romaji, pinyin, etc.).  Empty string when
        not applicable.
    part_of_speech:
        Normalised part-of-speech category string (e.g. ``"動詞"``, ``"名詞"``).
    extra:
        Engine-specific extra attributes (conjugation type, etc.).
    """

    surface: str
    base_form: str
    reading: str = ""
    part_of_speech: str = ""
    extra: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# BaseAligner
# ─────────────────────────────────────────────────────────────────────────────

class BaseAligner(ABC):
    """
    Abstract contract for bilingual subtitle alignment engines.

    Implementors must be stateless: the same instance must be safely reusable
    across multiple ``parse``/``align`` call sequences.
    """

    @abstractmethod
    def parse(self, raw: str) -> list[SubtitleBlock]:
        """
        Parse raw SRT content into an ordered list of subtitle blocks.

        Parameters
        ----------
        raw:
            Raw SRT file content (decoded string, any line ending).

        Returns
        -------
        list[SubtitleBlock]
            Ordered list of parsed subtitle blocks.

        Raises
        ------
        core.exceptions.SRTParseError
            If the content is structurally invalid and cannot be repaired.
        """

    @abstractmethod
    def align(
        self,
        source_blocks: Sequence[SubtitleBlock],
        target_blocks: Sequence[SubtitleBlock],
    ) -> list[AlignedPair]:
        """
        Align source and target subtitle blocks into matched pairs.

        Parameters
        ----------
        source_blocks:
            Parsed source-language subtitle blocks.
        target_blocks:
            Parsed target-language subtitle blocks.

        Returns
        -------
        list[AlignedPair]
            One entry per source block; ``AlignedPair.target`` is ``None``
            when no matching target block could be found.

        Raises
        ------
        core.exceptions.AlignmentError
            If alignment fails irrecoverably.
        """


# ─────────────────────────────────────────────────────────────────────────────
# BaseTokenizer
# ─────────────────────────────────────────────────────────────────────────────

class BaseTokenizer(ABC):
    """
    Abstract contract for language-specific tokenisation engines.

    Implementations are expected to be **lazily initialised**: heavyweight
    model loading should occur in :meth:`__init__` so that callers can catch
    :class:`core.exceptions.EngineInitialisationError` early.
    """

    #: BCP-47 language code this tokeniser handles (e.g. ``"ja"``, ``"zh"``).
    lang_code: str = ""

    @abstractmethod
    def tokenize(self, text: str) -> list[Token]:
        """
        Tokenise ``text`` into an ordered list of :class:`Token` objects.

        Parameters
        ----------
        text:
            Input string in the language this tokeniser handles.

        Returns
        -------
        list[Token]
            Ordered list of tokens extracted from ``text``.

        Raises
        ------
        core.exceptions.TokenizationError
            If tokenisation fails for any reason.
        """

    def tokenize_stream(self, texts: Sequence[str]) -> Iterator[list[Token]]:
        """
        Tokenise a sequence of texts, yielding token lists one at a time.

        Default implementation calls :meth:`tokenize` for each item.
        Subclasses may override for batch efficiency.

        Parameters
        ----------
        texts:
            Iterable of input strings.

        Yields
        ------
        list[Token]
            Token list for each input string.
        """
        for text in texts:
            yield self.tokenize(text)
