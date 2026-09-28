"""
tests/test_ui_app.py
~~~~~~~~~~~~~~~~~~~~
Integration tests for Streamlit app UI interactions and pure Japanese SRT translation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from app import _parse_and_translate_ja, _inject_css
from core.base import SubtitleBlock, AlignedPair


class TestPureJapaneseSRTTranslation:
    """Test pure Japanese SRT upload and translation flow."""

    def test_parse_and_translate_ja_helper(self) -> None:
        ja_srt = """1
00:00:01,000 --> 00:00:03,500
こんにちは世界。
"""
        with patch("app._init_transcriber") as mock_init_t:
            mock_t = MagicMock()
            mock_t.translate_subtitle_blocks.return_value = [
                (
                    SubtitleBlock(index=1, start_ms=1000, end_ms=3500, text="こんにちは世界。"),
                    SubtitleBlock(index=1, start_ms=1000, end_ms=3500, text="你好世界。"),
                )
            ]
            mock_init_t.return_value = mock_t

            _parse_and_translate_ja(ja_srt.encode("utf-8"))

    def test_apptest_pure_ja_srt_workflow(self) -> None:
        app_path = Path(__file__).parents[1] / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()
        assert not at.exception

        ja_content = b"""1
00:00:01,000 --> 00:00:03,500
\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e\xe3\x81\xae\xe5\x8b\x89\xe5\xbc\xb7\xe3\x80\x82
"""
        ja_uploader = at.file_uploader(key="ja_uploader")
        ja_uploader.upload(filename="test_ja.srt", content=ja_content)
        at.run()
        assert not at.exception

        btn_parse = [b for b in at.button if b.key == "btn_srt_parse"][0]
        assert btn_parse.disabled is True

        btn_tr = [b for b in at.button if b.key == "btn_ja_auto_translate"][0]
        assert btn_tr.disabled is False

        with patch("core.asr.transcriber.AudioTranscriber.translate_batch") as mock_batch:
            mock_batch.return_value = ["日語的學習。"]
            btn_tr.click().run()

        assert not at.exception
        pairs = at.session_state["aligned_pairs"]
        assert len(pairs) == 1
        assert pairs[0].target.text == "日語的學習。"


class TestTokenChipsInteraction:
    """Test dynamic-width token chips and click-to-save interaction."""

    def test_token_chip_toggle_to_saved(self) -> None:
        app_path = Path(__file__).parents[1] / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()

        p = AlignedPair(
            source=SubtitleBlock(index=1, start_ms=1000, end_ms=3500, text="美味しい寿司。"),
            target=SubtitleBlock(index=1, start_ms=1000, end_ms=3500, text="美味的壽司。"),
            overlap_ratio=1.0,
        )
        at.session_state["aligned_pairs"] = [p]
        at.session_state["saved_surfaces"] = set()
        at.run()

        token_btns = [b for b in at.button if b.key and b.key.startswith("tok_")]
        assert len(token_btns) >= 2

        # Initially unselected: type is secondary, no checkmark
        for b in token_btns:
            assert b.proto.type == "secondary"
            assert not b.label.startswith("✓")

        # Click the '寿司' token button
        sushi_btn = [b for b in token_btns if "寿司" in b.label][0]
        sushi_btn.click().run()
        assert not at.exception

        # After click: '寿司' button must be primary and have checkmark
        token_btns_after = [b for b in at.button if b.key and b.key.startswith("tok_")]
        sushi_after = [b for b in token_btns_after if "寿司" in b.label][0]
        assert sushi_after.proto.type == "primary"
        assert sushi_after.label.startswith("✓")


class TestStylesheetInjection:
    """Test external stylesheet loading from assets/style.css."""

    def test_inject_css_loads_stylesheet(self) -> None:
        css_file = Path(__file__).parents[1] / "assets" / "style.css"
        assert css_file.exists(), "assets/style.css must exist"
        content = css_file.read_text(encoding="utf-8")
        assert 'div[data-testid="stHorizontalBlock"]:has(button)' in content
        assert "font-size: 0.96rem" in content
        assert "line-height: 1.65" in content
        assert "color: #e2e8f0" in content

        with patch("streamlit.markdown") as mock_markdown:
            _inject_css()
            mock_markdown.assert_called_once()
            args, kwargs = mock_markdown.call_args
            assert "<style>" in args[0]
            assert 'div[data-testid="stHorizontalBlock"]:has(button)' in args[0]
            assert kwargs.get("unsafe_allow_html") is True

