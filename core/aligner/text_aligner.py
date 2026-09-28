"""
core/aligner/text_aligner.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sliding-window bilingual subtitle sentence-level alignment.

Algorithm overview
------------------
For every source block *s*, the aligner searches a **sliding window** of
candidate target blocks whose timestamps overlap with *s*.  The best match is
chosen by maximising the **temporal overlap ratio** defined as::

    overlap_ratio = intersection_duration / union_duration

where::

    intersection_duration = max(0, min(s.end, t.end) - max(s.start, t.start))
    union_duration        = max(s.end, t.end) - min(s.start, t.start)

If the best overlap ratio is below :attr:`TextAligner.min_overlap_ratio` the
source block is emitted as an unmatched pair (``AlignedPair.target is None``).

Window management
~~~~~~~~~~~~~~~~~
To avoid O(N·M) complexity the aligner maintains a *window start pointer* that
advances forward as source blocks advance in time.  Because real SRT files are
nearly sorted, this keeps average complexity close to O(N + M).

The window half-width (``window_ms``, default 5 s) must be at least as large
as the maximum expected timing drift between the two tracks.
"""

from __future__ import annotations

import logging
from typing import Sequence

from core.base import AlignedPair, BaseAligner, SubtitleBlock

logger = logging.getLogger(__name__)


def _overlap_ratio(a: SubtitleBlock, b: SubtitleBlock) -> float:
    """
    Compute the temporal Jaccard overlap ratio between two subtitle blocks.

    Returns a value in ``[0.0, 1.0]``:
    - ``1.0`` means perfect overlap (identical start/end).
    - ``0.0`` means no overlap at all.
    """
    intersection = max(0, min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms))
    if intersection == 0:
        return 0.0
    union = max(a.end_ms, b.end_ms) - min(a.start_ms, b.start_ms)
    return intersection / union if union > 0 else 0.0


class TextAligner(BaseAligner):
    """
    Sliding-window bilingual subtitle aligner.

    Parameters
    ----------
    window_ms:
        Half-width of the search window in milliseconds (default 5000 ms).
        Blocks whose start or end times fall within ``source.start - window_ms``
        to ``source.end + window_ms`` are considered candidates.
    min_overlap_ratio:
        Minimum Jaccard overlap ratio required to accept a match (default 0.1).
        Source blocks with no candidate exceeding this threshold are emitted as
        unmatched (``AlignedPair.target is None``).
    """

    def __init__(
        self,
        window_ms: int = 5_000,
        min_overlap_ratio: float = 0.1,
    ) -> None:
        if window_ms < 0:
            raise ValueError("window_ms must be non-negative.")
        if not (0.0 < min_overlap_ratio <= 1.0):
            raise ValueError("min_overlap_ratio must be in (0.0, 1.0].")
        self.window_ms = window_ms
        self.min_overlap_ratio = min_overlap_ratio

    # ── BaseAligner.parse() ─────────────────────────────────────────────────

    def parse(self, raw: str) -> list[SubtitleBlock]:
        """
        Not the primary purpose of this class; delegates to SRTParser.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "TextAligner does not parse SRT text; use SRTParser.parse() instead."
        )

    # ── BaseAligner.align() ─────────────────────────────────────────────────

    def align(
        self,
        source_blocks: Sequence[SubtitleBlock],
        target_blocks: Sequence[SubtitleBlock],
    ) -> list[AlignedPair]:
        """
        Align source blocks to target blocks using the sliding-window algorithm.

        Parameters
        ----------
        source_blocks:
            Parsed source-language subtitle blocks (need not be perfectly sorted).
        target_blocks:
            Parsed target-language subtitle blocks.

        Returns
        -------
        list[AlignedPair]
            One :class:`~core.base.AlignedPair` per source block.
        """
        if not source_blocks:
            return []
        if not target_blocks:
            logger.warning("Target track is empty; all source blocks are unmatched.")
            return [
                AlignedPair(source=s, target=None, overlap_ratio=0.0)
                for s in source_blocks
            ]

        # Work with sorted copies to keep the window pointer logic correct.
        src = sorted(source_blocks, key=lambda b: b.start_ms)
        tgt = sorted(target_blocks, key=lambda b: b.start_ms)

        pairs: list[AlignedPair] = []
        win_start = 0  # left boundary of the sliding window into tgt

        for s in src:
            # Advance window start: drop target blocks that ended well before s
            while (
                win_start < len(tgt)
                and tgt[win_start].end_ms < s.start_ms - self.window_ms
            ):
                win_start += 1

            # Collect candidate target blocks within the window
            best_block: SubtitleBlock | None = None
            best_ratio = 0.0

            for j in range(win_start, len(tgt)):
                t = tgt[j]
                # Stop scanning once we've passed the window's right edge
                if t.start_ms > s.end_ms + self.window_ms:
                    break
                ratio = _overlap_ratio(s, t)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_block = t

            if best_ratio >= self.min_overlap_ratio:
                pairs.append(
                    AlignedPair(source=s, target=best_block, overlap_ratio=best_ratio)
                )
                logger.debug(
                    "Matched block %d → %s  (overlap=%.3f)",
                    s.index,
                    best_block.index if best_block else "—",
                    best_ratio,
                )
            else:
                pairs.append(AlignedPair(source=s, target=None, overlap_ratio=0.0))
                logger.debug("No match for source block %d.", s.index)

        return pairs
