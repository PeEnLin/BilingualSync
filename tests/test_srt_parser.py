"""
tests/test_srt_parser.py
~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for :class:`~core.aligner.srt_parser.SRTParser` and
:class:`~core.aligner.text_aligner.TextAligner`.

Test coverage
-------------
``TestSRTParserBasic``
    Happy-path: standard SRT, multi-block, multi-line text.
``TestSRTParserTimestampFormats``
    Timestamp variant tolerance: period separator, no-millisecond, leading zeros.
``TestSRTParserFaultTolerance``
    BOM stripping, CRLF normalisation, tag stripping, corrupted block skipping.
``TestSRTParserEdgeCases``
    Empty input, no-block input, extra blank lines, single block.
``TestTextAlignerBasic``
    Perfect-overlap, partial-overlap, no-overlap, empty inputs.
``TestTextAlignerWindow``
    Window boundary conditions and min_overlap_ratio threshold.
"""

from __future__ import annotations

import pytest

from core.aligner.srt_parser import SRTParser
from core.aligner.text_aligner import TextAligner
from core.base import SubtitleBlock
from core.exceptions import MalformedTimestampError, SRTParseError


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _block(index: int, start_ms: int, end_ms: int, text: str = "text") -> SubtitleBlock:
    return SubtitleBlock(index=index, start_ms=start_ms, end_ms=end_ms, text=text)


def _srt(*blocks: tuple[int, str, str, str]) -> str:
    """
    Build an SRT string from tuples of (index, start_ts, end_ts, text).
    Timestamps should be in ``HH:MM:SS,mmm`` format.
    """
    parts = []
    for idx, start, end, text in blocks:
        parts.append(f"{idx}\n{start} --> {end}\n{text}\n")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def parser() -> SRTParser:
    return SRTParser()


@pytest.fixture()
def aligner() -> TextAligner:
    return TextAligner(window_ms=2000, min_overlap_ratio=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# TestSRTParserBasic
# ─────────────────────────────────────────────────────────────────────────────

class TestSRTParserBasic:
    """Happy-path parsing tests."""

    def test_single_block(self, parser: SRTParser) -> None:
        raw = "1\n00:00:01,000 --> 00:00:03,500\nHello world\n"
        blocks = parser.parse(raw)
        assert len(blocks) == 1
        b = blocks[0]
        assert b.index == 1
        assert b.start_ms == 1_000
        assert b.end_ms == 3_500
        assert b.text == "Hello world"

    def test_multiple_blocks(self, parser: SRTParser) -> None:
        raw = _srt(
            (1, "00:00:01,000", "00:00:03,000", "First"),
            (2, "00:00:04,000", "00:00:06,000", "Second"),
            (3, "00:00:07,000", "00:00:09,000", "Third"),
        )
        blocks = parser.parse(raw)
        assert len(blocks) == 3
        assert [b.index for b in blocks] == [1, 2, 3]
        assert [b.text for b in blocks] == ["First", "Second", "Third"]

    def test_multiline_text_joined(self, parser: SRTParser) -> None:
        raw = "1\n00:00:01,000 --> 00:00:04,000\nLine one\nLine two\n\n"
        blocks = parser.parse(raw)
        assert blocks[0].text == "Line one Line two"

    def test_start_end_ms_correct(self, parser: SRTParser) -> None:
        raw = "1\n01:02:03,456 --> 01:02:07,890\nTest\n"
        b = parser.parse(raw)[0]
        # 1*3600*1000 + 2*60*1000 + 3*1000 + 456 = 3_723_456
        assert b.start_ms == 3_723_456
        # 1*3600*1000 + 2*60*1000 + 7*1000 + 890 = 3_727_890
        assert b.end_ms == 3_727_890

    def test_duration_ms_property(self, parser: SRTParser) -> None:
        raw = "1\n00:00:01,000 --> 00:00:04,500\nHi\n"
        b = parser.parse(raw)[0]
        assert b.duration_ms == 3_500


# ─────────────────────────────────────────────────────────────────────────────
# TestSRTParserTimestampFormats
# ─────────────────────────────────────────────────────────────────────────────

class TestSRTParserTimestampFormats:
    """Timestamp format tolerance."""

    def test_period_separator(self, parser: SRTParser) -> None:
        raw = "1\n00:00:01.000 --> 00:00:03.500\nPeriod sep\n"
        b = parser.parse(raw)[0]
        assert b.start_ms == 1_000
        assert b.end_ms == 3_500

    def test_no_milliseconds(self, parser: SRTParser) -> None:
        raw = "1\n00:00:01 --> 00:00:03\nNo ms\n"
        b = parser.parse(raw)[0]
        assert b.start_ms == 1_000
        assert b.end_ms == 3_000

    def test_short_milliseconds_padded(self, parser: SRTParser) -> None:
        # "1" should be treated as 100 ms, "12" as 120 ms
        raw = "1\n00:00:00,1 --> 00:00:00,12\nShort ms\n"
        b = parser.parse(raw)[0]
        assert b.start_ms == 100
        assert b.end_ms == 120

    def test_leading_zero_hours(self, parser: SRTParser) -> None:
        raw = "1\n00:00:00,000 --> 00:00:00,500\nZero start\n"
        b = parser.parse(raw)[0]
        assert b.start_ms == 0
        assert b.end_ms == 500


# ─────────────────────────────────────────────────────────────────────────────
# TestSRTParserFaultTolerance
# ─────────────────────────────────────────────────────────────────────────────

class TestSRTParserFaultTolerance:
    """Robustness against common real-world SRT defects."""

    def test_crlf_line_endings(self, parser: SRTParser) -> None:
        raw = "1\r\n00:00:01,000 --> 00:00:02,000\r\nCRLF\r\n\r\n"
        blocks = parser.parse(raw)
        assert len(blocks) == 1
        assert blocks[0].text == "CRLF"

    def test_cr_only_line_endings(self, parser: SRTParser) -> None:
        raw = "1\r00:00:01,000 --> 00:00:02,000\rCR only\r\r"
        blocks = parser.parse(raw)
        assert len(blocks) == 1
        assert blocks[0].text == "CR only"

    def test_html_bold_tag_stripped(self, parser: SRTParser) -> None:
        raw = "1\n00:00:01,000 --> 00:00:02,000\n<b>Bold text</b>\n"
        b = parser.parse(raw)[0]
        assert b.text == "Bold text"

    def test_html_italic_tag_stripped(self, parser: SRTParser) -> None:
        raw = "1\n00:00:01,000 --> 00:00:02,000\n<i>Italic</i>\n"
        b = parser.parse(raw)[0]
        assert b.text == "Italic"

    def test_ass_override_tag_stripped(self, parser: SRTParser) -> None:
        raw = "1\n00:00:01,000 --> 00:00:02,000\n{\\an8}Top title\n"
        b = parser.parse(raw)[0]
        assert b.text == "Top title"

    def test_corrupted_block_skipped_valid_block_retained(self, parser: SRTParser) -> None:
        """A block with a garbage index line should be skipped; valid blocks pass."""
        raw = (
            "1\n00:00:01,000 --> 00:00:02,000\nGood block\n\n"
            "GARBAGE_LINE\n00:00:03,000 --> 00:00:04,000\nSkipped\n\n"
            "2\n00:00:05,000 --> 00:00:06,000\nAlso good\n"
        )
        blocks = parser.parse(raw)
        # The corrupted block is skipped; 2 valid blocks remain
        assert len(blocks) == 2
        assert blocks[0].text == "Good block"
        assert blocks[1].text == "Also good"

    def test_extra_blank_lines_between_blocks(self, parser: SRTParser) -> None:
        raw = "1\n00:00:01,000 --> 00:00:02,000\nA\n\n\n\n2\n00:00:03,000 --> 00:00:04,000\nB\n"
        blocks = parser.parse(raw)
        assert len(blocks) == 2

    def test_utf8_bom_stripped_by_parse_file(self, parser: SRTParser, tmp_path) -> None:
        content = "1\n00:00:01,000 --> 00:00:02,000\nBOM test\n"
        bom_file = tmp_path / "bom.srt"
        bom_file.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
        blocks = parser.parse_file(bom_file)
        assert blocks[0].text == "BOM test"

    def test_japanese_srt_parsed_correctly(self, parser: SRTParser) -> None:
        raw = "1\n00:00:01,000 --> 00:00:03,000\n食べました\n"
        b = parser.parse(raw)[0]
        assert b.text == "食べました"


# ─────────────────────────────────────────────────────────────────────────────
# TestSRTParserEdgeCases
# ─────────────────────────────────────────────────────────────────────────────

class TestSRTParserEdgeCases:
    """Boundary conditions."""

    def test_empty_string_raises(self, parser: SRTParser) -> None:
        with pytest.raises(SRTParseError):
            parser.parse("")

    def test_whitespace_only_raises(self, parser: SRTParser) -> None:
        with pytest.raises(SRTParseError):
            parser.parse("   \n\n   ")

    def test_no_valid_blocks_raises(self, parser: SRTParser) -> None:
        with pytest.raises(SRTParseError):
            parser.parse("This is not an SRT file at all.\n")

    def test_empty_text_block_skipped(self, parser: SRTParser) -> None:
        """A block whose only text lines consist purely of tags (stripped to empty) is skipped."""
        raw = "1\n00:00:01,000 --> 00:00:02,000\n<b></b>\n\n2\n00:00:03,000 --> 00:00:04,000\nOK\n"
        blocks = parser.parse(raw)
        assert len(blocks) == 1
        assert blocks[0].text == "OK"


# ─────────────────────────────────────────────────────────────────────────────
# TestTextAlignerBasic
# ─────────────────────────────────────────────────────────────────────────────

class TestTextAlignerBasic:
    """Core alignment algorithm tests."""

    def test_perfect_overlap_matched(self, aligner: TextAligner) -> None:
        src = [_block(1, 1_000, 3_000, "Hello")]
        tgt = [_block(1, 1_000, 3_000, "你好")]
        pairs = aligner.align(src, tgt)
        assert len(pairs) == 1
        assert pairs[0].target is not None
        assert pytest.approx(pairs[0].overlap_ratio) == 1.0

    def test_partial_overlap_matched(self, aligner: TextAligner) -> None:
        src = [_block(1, 1_000, 4_000)]
        tgt = [_block(1, 2_000, 5_000)]
        pairs = aligner.align(src, tgt)
        assert pairs[0].target is not None
        # intersection=2000, union=4000 → 0.5
        assert pytest.approx(pairs[0].overlap_ratio, abs=1e-3) == 0.5

    def test_no_overlap_unmatched(self) -> None:
        aligner = TextAligner(window_ms=500, min_overlap_ratio=0.1)
        src = [_block(1, 1_000, 2_000)]
        tgt = [_block(1, 10_000, 12_000)]
        pairs = aligner.align(src, tgt)
        assert pairs[0].target is None
        assert pairs[0].overlap_ratio == 0.0

    def test_empty_source_returns_empty(self, aligner: TextAligner) -> None:
        pairs = aligner.align([], [_block(1, 0, 1000)])
        assert pairs == []

    def test_empty_target_all_unmatched(self, aligner: TextAligner) -> None:
        src = [_block(1, 0, 1000), _block(2, 2000, 3000)]
        pairs = aligner.align(src, [])
        assert len(pairs) == 2
        assert all(p.target is None for p in pairs)

    def test_multiple_candidates_best_chosen(self, aligner: TextAligner) -> None:
        src = [_block(1, 2_000, 5_000)]
        # t1 overlaps 1s, t2 overlaps 3s → t2 should win
        t1 = _block(1, 1_000, 3_000)
        t2 = _block(2, 2_000, 5_000)
        pairs = aligner.align(src, [t1, t2])
        assert pairs[0].target.index == 2  # type: ignore[union-attr]

    def test_result_count_equals_source_count(self, aligner: TextAligner) -> None:
        src = [_block(i, i * 2000, i * 2000 + 1500) for i in range(1, 6)]
        tgt = [_block(i, i * 2000, i * 2000 + 1500) for i in range(1, 6)]
        pairs = aligner.align(src, tgt)
        assert len(pairs) == len(src)


# ─────────────────────────────────────────────────────────────────────────────
# TestTextAlignerWindow
# ─────────────────────────────────────────────────────────────────────────────

class TestTextAlignerWindow:
    """Window and threshold boundary tests."""

    def test_min_overlap_ratio_threshold_respected(self) -> None:
        # overlap=0.1 exactly → just below 0.2 threshold → unmatched
        aligner = TextAligner(window_ms=5000, min_overlap_ratio=0.2)
        # src=[0,10000], tgt=[9000,10000] → intersection=1000, union=10000 → 0.1
        src = [_block(1, 0, 10_000)]
        tgt = [_block(1, 9_000, 10_000)]
        pairs = aligner.align(src, tgt)
        assert pairs[0].target is None

    def test_overlap_above_threshold_matched(self) -> None:
        aligner = TextAligner(window_ms=5000, min_overlap_ratio=0.05)
        src = [_block(1, 0, 10_000)]
        tgt = [_block(1, 9_000, 10_000)]
        pairs = aligner.align(src, tgt)
        assert pairs[0].target is not None

    def test_invalid_window_ms_raises(self) -> None:
        with pytest.raises(ValueError):
            TextAligner(window_ms=-1)

    def test_invalid_min_overlap_ratio_raises(self) -> None:
        with pytest.raises(ValueError):
            TextAligner(min_overlap_ratio=0.0)

    def test_many_source_many_target(self) -> None:
        """Ensure O(N+M) window approach produces correct results at scale."""
        aligner = TextAligner(window_ms=500, min_overlap_ratio=0.5)
        n = 50
        src = [_block(i, i * 2000, i * 2000 + 1800) for i in range(n)]
        tgt = [_block(i, i * 2000, i * 2000 + 1800) for i in range(n)]
        pairs = aligner.align(src, tgt)
        assert len(pairs) == n
        assert all(p.target is not None for p in pairs)
        # Each source block should be matched to the target block at the same offset
        for p in pairs:
            assert p.source.index == p.target.index  # type: ignore[union-attr]
