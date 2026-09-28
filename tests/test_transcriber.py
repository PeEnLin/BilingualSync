"""
tests/test_transcriber.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for core.asr.transcriber.AudioTranscriber.

The transcriber now uses a batch translation strategy:
  - Phase 1: Collect all ASR segments (no translation during ASR loop)
  - Phase 2: Single HTTP request to translate all sentences at once

Tests cover the new batch paths as well as backward-compatible single-sentence
translation used by the pure-SRT upload workflow.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from core.asr.transcriber import (
    AudioTranscriber,
    _SEP,
    _CHUNK_SIZE,
    _CHUNK_CHAR_LIMIT,
    _FALLBACK_MSG,
)
from core.base import SubtitleBlock


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation & model caching
# ─────────────────────────────────────────────────────────────────────────────

class TestAudioTranscriberInit:
    """Initialization and model caching tests."""

    def test_default_init(self) -> None:
        transcriber = AudioTranscriber()
        assert transcriber._default_model_size == "medium"
        assert transcriber._device == "auto"
        assert transcriber._compute_type == "int8"
        assert transcriber._cpu_threads == 4
        assert transcriber._models == {}

    def test_custom_init(self) -> None:
        transcriber = AudioTranscriber(
            default_model_size="small",
            device="cpu",
            compute_type="float32",
            cpu_threads=2,
        )
        assert transcriber._default_model_size == "small"
        assert transcriber._device == "cpu"
        assert transcriber._compute_type == "float32"
        assert transcriber._cpu_threads == 2

    @patch("faster_whisper.WhisperModel")
    def test_get_model_cached(self, mock_whisper_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_whisper_cls.return_value = mock_instance

        transcriber = AudioTranscriber(default_model_size="small")
        m1 = transcriber.get_model()
        m2 = transcriber.get_model("small")

        assert m1 is m2
        assert mock_whisper_cls.call_count == 1

    @patch("faster_whisper.WhisperModel")
    def test_get_model_fallback_on_init_failure(self, mock_whisper_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        # Primary call fails → fallback to cpu/float32 succeeds
        mock_whisper_cls.side_effect = [RuntimeError("int8 not supported"), mock_instance]

        transcriber = AudioTranscriber(default_model_size="small")
        m = transcriber.get_model()
        assert m is mock_instance
        assert mock_whisper_cls.call_count == 2
        # Verify fallback call uses cpu + float32
        mock_whisper_cls.assert_called_with(
            "small", device="cpu", compute_type="float32", cpu_threads=4
        )


# ─────────────────────────────────────────────────────────────────────────────
# Batch translation
# ─────────────────────────────────────────────────────────────────────────────

class TestTranslateBatch:
    """Tests for the chunked batch translation path."""

    @patch("deep_translator.GoogleTranslator")
    def test_batch_via_deep_translator_success(
        self, mock_cls: MagicMock
    ) -> None:
        """deep_translator successfully splits sentences via sentinel."""
        texts = ["こんにちは", "ありがとう"]
        joined_result = "你好" + _SEP.strip() + "謝謝"
        mock_cls.return_value.translate.return_value = joined_result

        transcriber = AudioTranscriber()
        result = transcriber.translate_batch(texts)

        assert result == ["你好", "謝謝"]
        # Only ONE translate call per chunk (batch within chunk)
        mock_cls.return_value.translate.assert_called_once()

    @patch("deep_translator.MyMemoryTranslator")
    @patch("deep_translator.GoogleTranslator")
    def test_batch_falls_back_to_mymemory_on_google_429(
        self,
        mock_dt_cls: MagicMock,
        mock_mm_cls: MagicMock,
    ) -> None:
        """When GoogleTranslator raises 429, MyMemory is automatically used as fallback."""
        mock_dt_cls.return_value.translate.side_effect = Exception("429 Too Many Requests")
        mock_mm_instance = MagicMock()
        mock_mm_instance.translate.side_effect = lambda t: "早安" if "おはよう" in t else "再見"
        mock_mm_cls.return_value = mock_mm_instance

        texts = ["おはよう", "さようなら"]
        transcriber = AudioTranscriber()
        result = transcriber.translate_batch(texts)

        assert result == ["早安", "再見"]
        assert mock_mm_instance.translate.called

    @patch("deep_translator.MyMemoryTranslator")
    @patch("deep_translator.GoogleTranslator")
    def test_batch_all_fail_returns_busy_message(
        self,
        mock_dt_cls: MagicMock,
        mock_mm_cls: MagicMock,
    ) -> None:
        """When all providers fail, the busy message is assigned instead of Japanese original."""
        mock_dt_cls.return_value.translate.side_effect = Exception("network error")
        mock_mm_cls.return_value.translate.side_effect = Exception("rate limit")

        texts = ["テスト文章A", "テスト文章B"]
        transcriber = AudioTranscriber()
        result = transcriber.translate_batch(texts)

        assert result == [_FALLBACK_MSG, _FALLBACK_MSG]

    def test_batch_empty_input(self) -> None:
        transcriber = AudioTranscriber()
        assert transcriber.translate_batch([]) == []

    @patch("deep_translator.GoogleTranslator")
    def test_batch_skips_symbol_only_segments(self, mock_cls: MagicMock) -> None:
        """Pure punctuation / symbol strings are passed through without API call."""
        texts = ["…", "おはよう", "…"]
        mock_cls.return_value.translate.return_value = "おはよう_result"
        transcriber = AudioTranscriber()
        result = transcriber.translate_batch(texts)

        # Symbol-only slots remain unchanged, only the real text is translated
        assert result[0] == "…"   # symbol: pass-through
        assert result[2] == "…"   # symbol: pass-through

    @patch("deep_translator.GoogleTranslator")
    def test_batch_chunks_large_input(self, mock_cls: MagicMock) -> None:
        """Input larger than _CHUNK_SIZE is split into multiple chunks."""
        # Build 65 sentences (>2 chunks of 30)
        texts = [f"日本語文 {i}" for i in range(65)]
        # For each chunk call, return joined Chinese result
        def side_effect(joined: str) -> str:
            parts = joined.split("|||SEP|||".strip())
            # Return plausible Chinese for each part
            return _SEP.strip().join(f"中文{i}" for i in range(len(parts)))

        mock_cls.return_value.translate.side_effect = lambda t: _SEP.strip().join(
            f"中文" for _ in t.split(_SEP.strip())
        )
        # We just verify no crash and correct count returned
        result = transcriber = AudioTranscriber()
        out = transcriber.translate_batch(texts)
        assert len(out) == 65


# ─────────────────────────────────────────────────────────────────────────────
# Single-sentence translate_text (SRT-upload backward compat)
# ─────────────────────────────────────────────────────────────────────────────

class TestTranslateText:
    """Backward-compatible single-sentence translate_text tests."""

    def test_empty_input(self) -> None:
        transcriber = AudioTranscriber()
        assert transcriber.translate_text("") == ""
        assert transcriber.translate_text("   \n\t  ") == ""

    @patch("deep_translator.MyMemoryTranslator")
    @patch("deep_translator.GoogleTranslator")
    def test_all_fail_returns_busy_message(
        self,
        mock_dt_cls: MagicMock,
        mock_mm_cls: MagicMock,
    ) -> None:
        """On total failure the busy placeholder message is returned."""
        mock_dt_cls.return_value.translate.side_effect = Exception("429")
        mock_mm_cls.return_value.translate.side_effect = Exception("network error")

        transcriber = AudioTranscriber()
        result = transcriber.translate_text("日本語のテスト")
        assert result == _FALLBACK_MSG

    @patch("deep_translator.MyMemoryTranslator")
    @patch("deep_translator.GoogleTranslator")
    def test_secondary_mymemory_fallback_success(
        self,
        mock_dt_cls: MagicMock,
        mock_mm_cls: MagicMock,
    ) -> None:
        """When Google raises 429, MyMemory recovers the translation."""
        mock_dt_cls.return_value.translate.side_effect = Exception("TooManyRequests 429")
        mock_mm_instance = MagicMock()
        mock_mm_instance.translate.return_value = "你好世界"
        mock_mm_cls.return_value = mock_mm_instance

        transcriber = AudioTranscriber()
        result = transcriber.translate_text("こんにちは世界")
        assert result == "你好世界"
        assert mock_mm_instance.translate.called


# ─────────────────────────────────────────────────────────────────────────────
# translate_subtitle_blocks (pure SRT path)
# ─────────────────────────────────────────────────────────────────────────────

class TestTranslateSubtitleBlocks:
    @patch("core.asr.transcriber.AudioTranscriber.translate_batch")
    def test_translate_subtitle_blocks(self, mock_batch: MagicMock) -> None:
        mock_batch.return_value = ["你好", "早安"]
        blocks = [
            SubtitleBlock(index=1, start_ms=0, end_ms=1000, text="こんにちは"),
            SubtitleBlock(index=2, start_ms=1200, end_ms=2500, text="おはよう"),
        ]
        transcriber = AudioTranscriber()
        pairs = transcriber.translate_subtitle_blocks(blocks, source="ja", target="zh-TW")

        assert len(pairs) == 2
        assert pairs[0][0].text == "こんにちは"
        assert pairs[0][1].text == "你好"
        assert pairs[0][1].start_ms == 0
        assert pairs[0][1].end_ms == 1000
        assert pairs[1][0].text == "おはよう"
        assert pairs[1][1].text == "早安"

        # translate_batch called ONCE with all texts
        mock_batch.assert_called_once_with(
            ["こんにちは", "おはよう"], source="ja", target="zh-TW"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Full transcribe_and_translate pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestAudioTranscriberPipeline:
    """Integration tests for the full ASR + batch-translate pipeline."""

    def test_nonexistent_file_raises_filenotfound(self, tmp_path: Path) -> None:
        transcriber = AudioTranscriber()
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe_and_translate(tmp_path / "does_not_exist.mp4")

    @patch("core.asr.transcriber.AudioTranscriber.translate_batch")
    @patch("core.asr.transcriber.AudioTranscriber.get_model")
    def test_transcribe_and_translate_success(
        self,
        mock_get_model: MagicMock,
        mock_batch: MagicMock,
        tmp_path: Path,
    ) -> None:
        media_file = tmp_path / "sample.mp3"
        media_file.write_bytes(b"dummy audio content")

        seg1 = MagicMock()
        seg1.start = 1.0
        seg1.end = 3.5
        seg1.text = " 昨日は美味しいお寿司を食べました。 "

        seg2 = MagicMock()
        seg2.start = 4.0
        seg2.end = 6.2
        seg2.text = " とても楽しかったです。 "

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], None)
        mock_get_model.return_value = mock_model

        # Batch translate returns two results in one call
        mock_batch.return_value = ["昨天吃了美味的壽司。", "非常開心。"]

        transcriber = AudioTranscriber(default_model_size="medium")
        pairs = transcriber.transcribe_and_translate(media_file)

        assert len(pairs) == 2

        ja_b1, zh_b1 = pairs[0]
        assert isinstance(ja_b1, SubtitleBlock)
        assert isinstance(zh_b1, SubtitleBlock)
        assert ja_b1.index == 1
        assert ja_b1.start_ms == 1000
        assert ja_b1.end_ms == 3500
        assert ja_b1.text == "昨日は美味しいお寿司を食べました。"
        assert zh_b1.text == "昨天吃了美味的壽司。"

        ja_b2, zh_b2 = pairs[1]
        assert ja_b2.index == 2
        assert ja_b2.start_ms == 4000
        assert ja_b2.end_ms == 6200
        assert ja_b2.text == "とても楽しかったです。"
        assert zh_b2.text == "非常開心。"

        # translate_batch called ONCE after all ASR is done (batch, not per-sentence)
        mock_batch.assert_called_once_with(
            ["昨日は美味しいお寿司を食べました。", "とても楽しかったです。"],
            source="ja",
            target="zh-TW",
        )

        # Verify ASR parameters
        mock_model.transcribe.assert_called_once_with(
            str(media_file),
            language="ja",
            beam_size=3,
            task="transcribe",
            initial_prompt="こんにちは。日本語の字幕です。",
            condition_on_previous_text=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 700, "speech_pad_ms": 300},
        )

    @patch("core.asr.transcriber.AudioTranscriber.translate_batch")
    @patch("core.asr.transcriber.AudioTranscriber.get_model")
    def test_transcription_translation_fallback_on_network_error(
        self,
        mock_get_model: MagicMock,
        mock_batch: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When batch translate fails, source text is preserved – no empty column."""
        media_file = tmp_path / "sample.mp4"
        media_file.write_bytes(b"dummy video content")

        seg = MagicMock()
        seg.start = 0.5
        seg.end = 2.0
        seg.text = "こんにちは"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], None)
        mock_get_model.return_value = mock_model

        # translate_batch returns original on failure
        mock_batch.return_value = ["こんにちは"]

        transcriber = AudioTranscriber()
        pairs = transcriber.transcribe_and_translate(media_file)

        assert len(pairs) == 1
        ja_b, zh_b = pairs[0]
        assert ja_b.text == "こんにちは"
        # Source preserved – subtitle column never blank
        assert zh_b.text == "こんにちは"

    @patch("core.asr.transcriber.AudioTranscriber.translate_batch")
    @patch("core.asr.transcriber.AudioTranscriber.get_model")
    def test_empty_audio_returns_empty_list(
        self,
        mock_get_model: MagicMock,
        mock_batch: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Silent media with no recognised segments returns empty list."""
        media_file = tmp_path / "silent.mp3"
        media_file.write_bytes(b"silent audio")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], None)
        mock_get_model.return_value = mock_model

        transcriber = AudioTranscriber()
        pairs = transcriber.transcribe_and_translate(media_file)

        assert pairs == []
        mock_batch.assert_not_called()  # batch must not be called for empty results
