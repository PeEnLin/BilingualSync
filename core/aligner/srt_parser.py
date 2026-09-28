"""
core/aligner/srt_parser.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Streaming SRT subtitle parser with robust fault-tolerance.

Features
--------
* Strips UTF-8 / UTF-16 BOM automatically.
* Normalises CRLF, CR, and bare LF line endings uniformly.
* Tolerates missing blank lines between blocks (common in wild-captured SRTs).
* Skips and logs corrupted blocks rather than aborting the whole parse.
* Accepts multiple timestamp formats:
    - Standard:   ``HH:MM:SS,mmm --> HH:MM:SS,mmm``
    - Period sep: ``HH:MM:SS.mmm --> HH:MM:SS.mmm``
    - No ms:      ``HH:MM:SS --> HH:MM:SS``
* Collapses multi-line subtitle text into a single space-joined string.
* Strips common SRT formatting tags (``<b>``, ``<i>``, ``<u>``, ``{\\an8}``).

The parser exposes two entry points:

``SRTParser.parse(raw)``
    Parse a full in-memory string; returns a list of :class:`~core.base.SubtitleBlock`.

``SRTParser.parse_file(path)``
    Open, auto-detect encoding, and parse an SRT file from the filesystem.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Generator, Sequence

from core.base import BaseAligner, SubtitleBlock, AlignedPair
from core.exceptions import MalformedTimestampError, SRTParseError


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

# Matches  HH:MM:SS,mmm --> HH:MM:SS,mmm  (comma or period separator, ms optional)
_TS_PATTERN = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
    r"\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})",
)

# Also match timestamp lines without milliseconds: HH:MM:SS --> HH:MM:SS
_TS_NO_MS_PATTERN = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})"
    r"\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})",
)

# Strip common SRT inline formatting tags
_TAG_PATTERN = re.compile(
    r"<[^>]+>"           # HTML-style tags: <b>, <i>, <font color=…>, etc.
    r"|\{\\[^}]+\}"      # ASS/SSA override tags: {\an8}, {\pos(x,y)}, etc.
    r"|\u200b"           # zero-width space
    , re.IGNORECASE,
)

# BOMs to strip
_BOMS = (
    b"\xef\xbb\xbf",  # UTF-8
    b"\xff\xfe",       # UTF-16 LE
    b"\xfe\xff",       # UTF-16 BE
)


def _strip_bom(raw_bytes: bytes) -> bytes:
    for bom in _BOMS:
        if raw_bytes.startswith(bom):
            return raw_bytes[len(bom):]
    return raw_bytes


def _ms(h: str, m: str, s: str, ms: str) -> int:
    """Convert HH:MM:SS,mmm parts to milliseconds."""
    ms_val = ms.ljust(3, "0")[:3]  # normalise to exactly 3 digits
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms_val)


def _parse_timestamp_line(line: str, line_no: int) -> tuple[int, int] | None:
    """
    Return ``(start_ms, end_ms)`` from a timestamp line, or ``None`` if the
    line does not look like a timestamp at all.

    Raises :class:`~core.exceptions.MalformedTimestampError` if the line
    contains ``-->`` but cannot be fully parsed.
    """
    if "-->" not in line:
        return None

    m = _TS_PATTERN.search(line)
    if m:
        start = _ms(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _ms(m.group(5), m.group(6), m.group(7), m.group(8))
        return start, end

    m2 = _TS_NO_MS_PATTERN.search(line)
    if m2:
        start = _ms(m2.group(1), m2.group(2), m2.group(3), "0")
        end = _ms(m2.group(4), m2.group(5), m2.group(6), "0")
        return start, end

    raise MalformedTimestampError(raw_timestamp=line.strip(), line_number=line_no)


def _clean_text(text: str) -> str:
    """Strip formatting tags and normalise whitespace."""
    return _TAG_PATTERN.sub("", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Block iterator (generator)
# ─────────────────────────────────────────────────────────────────────────────

def _iter_blocks(lines: list[str]) -> Generator[SubtitleBlock, None, None]:
    """
    Iterate over ``lines`` and yield one :class:`SubtitleBlock` per SRT block.

    Corrupted / unrecognised blocks are skipped with a WARNING log entry.
    """
    i = 0
    n = len(lines)

    while i < n:
        # Skip blank lines between blocks
        while i < n and lines[i].strip() == "":
            i += 1
        if i >= n:
            break

        # ── 1. Index line ─────────────────────────────────────────────────
        index_line = lines[i].strip()
        if not index_line.isdigit():
            # Might be a corrupted block header; skip until next blank line
            logger.warning("Expected block index at line %d, got: %r", i + 1, index_line)
            while i < n and lines[i].strip() != "":
                i += 1
            continue
        block_index = int(index_line)
        i += 1

        # ── 2. Timestamp line ─────────────────────────────────────────────
        if i >= n:
            logger.warning("Block %d has no timestamp line; skipping.", block_index)
            break

        try:
            ts = _parse_timestamp_line(lines[i], i + 1)
        except MalformedTimestampError as exc:
            logger.warning("Skipping block %d: %s", block_index, exc.message)
            while i < n and lines[i].strip() != "":
                i += 1
            continue

        if ts is None:
            logger.warning(
                "Block %d: line %d does not look like a timestamp: %r",
                block_index, i + 1, lines[i],
            )
            while i < n and lines[i].strip() != "":
                i += 1
            continue

        start_ms, end_ms = ts
        i += 1

        # ── 3. Text lines ─────────────────────────────────────────────────
        text_parts: list[str] = []
        while i < n and lines[i].strip() != "":
            cleaned = _clean_text(lines[i])
            if cleaned:
                text_parts.append(cleaned)
            i += 1

        text = " ".join(text_parts)
        if not text:
            logger.debug("Block %d has empty text; skipping.", block_index)
            continue

        yield SubtitleBlock(
            index=block_index,
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public parser (implements BaseAligner.parse only)
# ─────────────────────────────────────────────────────────────────────────────

class SRTParser(BaseAligner):
    """
    Streaming SRT subtitle parser.

    This class implements :meth:`parse` from :class:`~core.base.BaseAligner`.
    The :meth:`align` method is intentionally delegated to
    :class:`~core.aligner.text_aligner.TextAligner`; calling it here raises
    ``NotImplementedError``.
    """

    # ── Encoding candidates tried in order ──────────────────────────────────
    _ENCODINGS = ("utf-8-sig", "utf-8", "shift-jis", "euc-jp", "cp950", "latin-1")

    def parse(self, raw: str) -> list[SubtitleBlock]:
        """
        Parse a raw SRT string into an ordered list of subtitle blocks.

        Parameters
        ----------
        raw:
            Decoded SRT content (any line ending).

        Returns
        -------
        list[SubtitleBlock]

        Raises
        ------
        SRTParseError
            If the content is empty or contains no parseable blocks.
        """
        if not raw or not raw.strip():
            raise SRTParseError("SRT content is empty.")

        # Normalise line endings
        normalised = raw.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalised.split("\n")

        blocks = list(_iter_blocks(lines))
        if not blocks:
            raise SRTParseError("No valid subtitle blocks found in the SRT content.")

        return blocks

    def parse_file(self, path: str | Path) -> list[SubtitleBlock]:
        """
        Open an SRT file, auto-detect encoding, and parse it.

        Parameters
        ----------
        path:
            Filesystem path to the ``.srt`` file.

        Returns
        -------
        list[SubtitleBlock]

        Raises
        ------
        SRTParseError
            If the file cannot be decoded with any known encoding.
        """
        path = Path(path)
        raw_bytes = _strip_bom(path.read_bytes())

        for enc in self._ENCODINGS:
            try:
                raw = raw_bytes.decode(enc)
                logger.debug("Decoded %s with encoding '%s'.", path.name, enc)
                return self.parse(raw)
            except (UnicodeDecodeError, LookupError):
                continue

        raise SRTParseError(
            f"Cannot decode SRT file with any known encoding: {path}",
            context={"file_path": str(path), "tried_encodings": list(self._ENCODINGS)},
        )

    # ── align() is not implemented in this class ────────────────────────────
    def align(
        self,
        source_blocks: Sequence[SubtitleBlock],
        target_blocks: Sequence[SubtitleBlock],
    ) -> list[AlignedPair]:
        raise NotImplementedError(
            "SRTParser does not implement align(); use TextAligner instead."
        )
