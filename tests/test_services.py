"""
tests/test_services.py
~~~~~~~~~~~~~~~~~~~~~~~
Unit and integration tests for the business service layer.

Test categories
---------------
``TestVocabularyServiceExtract``
    Happy-path and error-path for :meth:`VocabularyService.extract_and_save_word`.
``TestVocabularyServiceIdempotency``
    Duplicate-surface de-duplication (upsert semantics).
``TestVocabularyServiceSearch``
    :meth:`VocabularyService.search` substring matching.
``TestVocabularyServiceCRUD``
    :meth:`get_all_saved_words`, :meth:`delete_word`, :meth:`update_translation`,
    :meth:`update_notes`, :meth:`mark_reviewed`.
``TestVocabularyServiceNLPIntegration``
    Full-stack: real JapaneseTokenizer extracts correct features.
``TestExportServiceAnki``
    TSV content format: columns, delimiter, HTML escaping.
``TestExportServiceJson``
    JSON structure, key presence, example nesting.
``TestExportServiceCsv``
    CSV header and row count.
``TestExportServiceDispatch``
    :meth:`ExportService.export` format dispatch and unknown format error.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from core.base import BaseTokenizer, Token
from core.exceptions import RecordNotFoundError, UnsupportedExportFormatError
from database.connection import DatabaseConnection
from database.models import VocabularyEntry
from database.repository import VocabularyRepository
from services.export_service import ExportService
from services.vocab_service import VocabularyService


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures & helpers
# ─────────────────────────────────────────────────────────────────────────────

class _StubTokenizer(BaseTokenizer):
    """
    Deterministic stub tokenizer.

    Returns a single Token whose ``base_form``, ``reading``, and
    ``part_of_speech`` are predictable regardless of input.
    """
    lang_code = "ja"

    def tokenize(self, text: str) -> list[Token]:
        return [Token(
            surface=text,
            base_form=text + "_base",
            reading=text + "_reading",
            part_of_speech="動詞",
        )]


@pytest.fixture()
def repo():
    """In-memory VocabularyRepository, isolated per test."""
    with DatabaseConnection(db_path=Path(":memory:"), check_same_thread=True) as conn:
        yield VocabularyRepository(conn)


@pytest.fixture()
def stub_tok() -> _StubTokenizer:
    return _StubTokenizer()


@pytest.fixture()
def svc(repo, stub_tok) -> VocabularyService:
    return VocabularyService(repo=repo, tokenizer=stub_tok)


@pytest.fixture()
def export_svc(repo) -> ExportService:
    return ExportService(repo=repo, source_lang="ja")


def _seed(svc: VocabularyService, surface: str, **kwargs) -> VocabularyEntry:
    """Helper: save one word and return the persisted entry."""
    return svc.extract_and_save_word(surface, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# TestVocabularyServiceExtract
# ─────────────────────────────────────────────────────────────────────────────

class TestVocabularyServiceExtract:
    """extract_and_save_word happy-path and error cases."""

    def test_returns_vocabulary_entry(self, svc: VocabularyService) -> None:
        entry = svc.extract_and_save_word("食べた")
        assert isinstance(entry, VocabularyEntry)

    def test_entry_has_id(self, svc: VocabularyService) -> None:
        entry = svc.extract_and_save_word("走った")
        assert entry.id is not None
        assert entry.id > 0

    def test_surface_form_stored_correctly(self, svc: VocabularyService) -> None:
        entry = svc.extract_and_save_word("猫")
        assert entry.surface_form == "猫"

    def test_stub_base_form_applied(self, svc: VocabularyService) -> None:
        """Stub tokenizer returns surface + '_base' as base_form."""
        entry = svc.extract_and_save_word("犬")
        assert entry.base_form == "犬_base"

    def test_stub_reading_applied(self, svc: VocabularyService) -> None:
        entry = svc.extract_and_save_word("魚")
        assert entry.reading == "魚_reading"

    def test_stub_pos_applied(self, svc: VocabularyService) -> None:
        entry = svc.extract_and_save_word("行く")
        assert entry.part_of_speech == "動詞"

    def test_translation_stored(self, svc: VocabularyService) -> None:
        entry = svc.extract_and_save_word("猫", translation="cat")
        assert entry.translation == "cat"

    def test_notes_stored(self, svc: VocabularyService) -> None:
        entry = svc.extract_and_save_word("猫", notes="My mnemonic")
        assert entry.notes == "My mnemonic"

    def test_source_lang_default_ja(self, svc: VocabularyService) -> None:
        entry = svc.extract_and_save_word("猫")
        assert entry.source_lang == "ja"

    def test_target_lang_default_zh_tw(self, svc: VocabularyService) -> None:
        entry = svc.extract_and_save_word("猫")
        assert entry.target_lang == "zh-tw"

    def test_leading_trailing_whitespace_stripped(self, svc: VocabularyService) -> None:
        entry = svc.extract_and_save_word("  猫  ")
        assert entry.surface_form == "猫"

    def test_blank_input_raises_value_error(self, svc: VocabularyService) -> None:
        with pytest.raises(ValueError, match="empty"):
            svc.extract_and_save_word("   ")

    def test_example_sentence_attached(self, svc: VocabularyService, repo) -> None:
        entry = svc.extract_and_save_word(
            "食べた",
            source_sentence="昨日ラーメンを食べた。",
            target_sentence="昨天吃了拉麵。",
        )
        examples = repo.get_examples(entry.id)
        assert len(examples) == 1
        assert examples[0].source_text == "昨日ラーメンを食べた。"
        assert examples[0].target_text == "昨天吃了拉麵。"

    def test_example_with_timestamps_stored(self, svc: VocabularyService, repo) -> None:
        entry = svc.extract_and_save_word(
            "走る",
            source_sentence="毎朝走る。",
            source_start_ms=1000,
            source_end_ms=3000,
        )
        examples = repo.get_examples(entry.id)
        assert examples[0].source_start_ms == 1000
        assert examples[0].source_end_ms == 3000

    def test_no_example_when_source_sentence_empty(self, svc: VocabularyService, repo) -> None:
        entry = svc.extract_and_save_word("猫", source_sentence="")
        examples = repo.get_examples(entry.id)
        assert len(examples) == 0

    def test_tokenizer_failure_saves_surface_only(self, repo) -> None:
        """If the tokenizer raises, entry is saved with surface as base_form."""
        bad_tok = MagicMock(spec=BaseTokenizer)
        bad_tok.lang_code = "ja"
        from core.exceptions import TokenizationError
        bad_tok.tokenize.side_effect = TokenizationError("boom")
        svc = VocabularyService(repo=repo, tokenizer=bad_tok)
        entry = svc.extract_and_save_word("壊れた")
        assert entry.surface_form == "壊れた"
        assert entry.base_form == "壊れた"


# ─────────────────────────────────────────────────────────────────────────────
# TestVocabularyServiceIdempotency
# ─────────────────────────────────────────────────────────────────────────────

class TestVocabularyServiceIdempotency:
    """Duplicate surface de-duplication (upsert semantics)."""

    def test_duplicate_surface_returns_existing(self, svc: VocabularyService) -> None:
        e1 = svc.extract_and_save_word("猫")
        e2 = svc.extract_and_save_word("猫")
        assert e1.id == e2.id

    def test_word_count_unchanged_on_duplicate(self, svc: VocabularyService) -> None:
        svc.extract_and_save_word("猫")
        svc.extract_and_save_word("猫")
        assert svc.get_word_count() == 1

    def test_duplicate_still_appends_example(self, svc: VocabularyService, repo) -> None:
        """Even for existing entries, new example sentences are attached."""
        e1 = svc.extract_and_save_word("猫", source_sentence="猫がいる。")
        svc.extract_and_save_word("猫", source_sentence="猫が走る。")
        examples = repo.get_examples(e1.id)
        assert len(examples) == 2

    def test_different_surfaces_both_saved(self, svc: VocabularyService) -> None:
        svc.extract_and_save_word("猫")
        svc.extract_and_save_word("犬")
        assert svc.get_word_count() == 2

    def test_different_lang_pair_both_saved(self, repo, stub_tok) -> None:
        svc_ja_zh = VocabularyService(repo=repo, tokenizer=stub_tok, source_lang="ja", target_lang="zh-tw")
        svc_ja_en = VocabularyService(repo=repo, tokenizer=stub_tok, source_lang="ja", target_lang="en")
        svc_ja_zh.extract_and_save_word("猫")
        svc_ja_en.extract_and_save_word("猫")
        # Both entries exist (different target_lang)
        assert repo.count() == 2


# ─────────────────────────────────────────────────────────────────────────────
# TestVocabularyServiceSearch
# ─────────────────────────────────────────────────────────────────────────────

class TestVocabularyServiceSearch:
    """search() substring matching."""

    def setup_method(self) -> None:
        pass  # fixtures handle setup

    def test_search_by_surface(self, svc: VocabularyService) -> None:
        svc.extract_and_save_word("食べる", translation="eat")
        svc.extract_and_save_word("走る", translation="run")
        results = svc.search("食べ")
        assert len(results) == 1
        assert results[0].surface_form == "食べる"

    def test_search_by_translation(self, svc: VocabularyService) -> None:
        svc.extract_and_save_word("猫", translation="cat")
        svc.extract_and_save_word("犬", translation="dog")
        results = svc.search("cat")
        assert len(results) == 1
        assert results[0].surface_form == "猫"

    def test_search_case_insensitive(self, svc: VocabularyService) -> None:
        svc.extract_and_save_word("猫", translation="Cat")
        results = svc.search("CAT")
        assert len(results) == 1

    def test_search_empty_returns_all(self, svc: VocabularyService) -> None:
        svc.extract_and_save_word("猫")
        svc.extract_and_save_word("犬")
        results = svc.search("")
        assert len(results) == 2

    def test_search_no_match_returns_empty(self, svc: VocabularyService) -> None:
        svc.extract_and_save_word("猫")
        results = svc.search("xyz_no_match")
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# TestVocabularyServiceCRUD
# ─────────────────────────────────────────────────────────────────────────────

class TestVocabularyServiceCRUD:
    """get_all_saved_words, delete_word, update_translation, mark_reviewed."""

    def test_get_all_saved_words_returns_list(self, svc: VocabularyService) -> None:
        svc.extract_and_save_word("猫")
        result = svc.get_all_saved_words()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_get_all_saved_words_empty_initially(self, svc: VocabularyService) -> None:
        assert svc.get_all_saved_words() == []

    def test_delete_word_removes_entry(self, svc: VocabularyService) -> None:
        entry = svc.extract_and_save_word("猫")
        svc.delete_word(entry.id)
        assert svc.get_word_count() == 0

    def test_delete_all_words(self, svc: VocabularyService) -> None:
        svc.extract_and_save_word("猫")
        svc.extract_and_save_word("犬")
        assert svc.get_word_count() == 2
        count = svc.delete_all_words()
        assert count == 2
        assert svc.get_word_count() == 0

    def test_delete_nonexistent_raises(self, svc: VocabularyService) -> None:
        with pytest.raises(RecordNotFoundError):
            svc.delete_word(99999)

    def test_update_translation(self, svc: VocabularyService) -> None:
        entry = svc.extract_and_save_word("猫")
        updated = svc.update_translation(entry.id, "cat")
        assert updated.translation == "cat"

    def test_update_notes(self, svc: VocabularyService) -> None:
        entry = svc.extract_and_save_word("猫")
        updated = svc.update_notes(entry.id, "Note about cats")
        assert updated.notes == "Note about cats"

    def test_mark_reviewed_increments_count(self, svc: VocabularyService) -> None:
        entry = svc.extract_and_save_word("猫")
        assert entry.review_count == 0
        updated = svc.mark_reviewed(entry.id)
        assert updated.review_count == 1
        updated2 = svc.mark_reviewed(entry.id)
        assert updated2.review_count == 2

    def test_get_word_count(self, svc: VocabularyService) -> None:
        assert svc.get_word_count() == 0
        svc.extract_and_save_word("猫")
        assert svc.get_word_count() == 1
        svc.extract_and_save_word("犬")
        assert svc.get_word_count() == 2

    def test_get_examples(self, svc: VocabularyService) -> None:
        entry = svc.extract_and_save_word("猫", source_sentence="猫がいる。")
        examples = svc.get_examples(entry.id)
        assert len(examples) == 1
        assert examples[0].source_text == "猫がいる。"


# ─────────────────────────────────────────────────────────────────────────────
# TestVocabularyServiceNLPIntegration
# ─────────────────────────────────────────────────────────────────────────────

class TestVocabularyServiceNLPIntegration:
    """Full-stack: real JapaneseTokenizer + real DB."""

    @pytest.fixture()
    def real_svc(self, repo) -> VocabularyService:
        from core.nlp.japanese_engine import JapaneseTokenizer
        return VocabularyService(repo=repo, tokenizer=JapaneseTokenizer())

    def test_past_tense_verb_base_form_restored(self, real_svc: VocabularyService) -> None:
        """食べた → base_form must be 食べる from Janome."""
        entry = real_svc.extract_and_save_word("食べた")
        # Janome splits 食べた → 食べ (base: 食べる) + た
        # _find_token returns first token for the stem "食べ"
        assert entry.base_form == "食べる"

    def test_verb_pos_is_doushi(self, real_svc: VocabularyService) -> None:
        entry = real_svc.extract_and_save_word("走った")
        assert entry.part_of_speech == "動詞"

    def test_noun_pos_is_meishi(self, real_svc: VocabularyService) -> None:
        entry = real_svc.extract_and_save_word("猫")
        assert entry.part_of_speech == "名詞"

    def test_reading_populated_for_kanji(self, real_svc: VocabularyService) -> None:
        entry = real_svc.extract_and_save_word("東京")
        # Katakana reading from Janome
        assert entry.reading  # should be "トウキョウ" or similar non-empty

    def test_exact_surface_preserved(self, real_svc: VocabularyService) -> None:
        entry = real_svc.extract_and_save_word("食べる")
        assert entry.surface_form == "食べる"
        assert entry.base_form == "食べる"


# ─────────────────────────────────────────────────────────────────────────────
# TestExportServiceAnki
# ─────────────────────────────────────────────────────────────────────────────

class TestExportServiceAnki:
    """Anki TSV export format validation."""

    def _populate(self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path):
        """Seed two entries; return the export file path."""
        svc.extract_and_save_word(
            "食べた",
            translation="ate",
            source_sentence="昨日ラーメンを食べた。",
            target_sentence="昨天吃了拉麵。",
        )
        svc.extract_and_save_word("猫", translation="cat")
        out = tmp_path / "anki_export.tsv"
        export_svc.export_to_anki(out)
        return out

    def test_row_count_equals_entry_count(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        out = self._populate(svc, export_svc, tmp_path)
        rows = out.read_text(encoding="utf-8-sig").strip().split("\n")
        assert len(rows) == 2

    def test_tab_delimiter(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        out = self._populate(svc, export_svc, tmp_path)
        first_line = out.read_text(encoding="utf-8-sig").split("\n")[0]
        assert "\t" in first_line

    def test_seven_columns(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        out = self._populate(svc, export_svc, tmp_path)
        reader = csv.reader(
            io.StringIO(out.read_text(encoding="utf-8-sig")), delimiter="\t"
        )
        for row in reader:
            assert len(row) == 7, f"Expected 7 columns, got {len(row)}: {row}"

    def test_surface_in_first_column(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        out = self._populate(svc, export_svc, tmp_path)
        surfaces = set()
        reader = csv.reader(
            io.StringIO(out.read_text(encoding="utf-8-sig")), delimiter="\t"
        )
        for row in reader:
            surfaces.add(row[0])
        assert "食べた" in surfaces
        assert "猫" in surfaces

    def test_example_sentence_in_column_4(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        out = self._populate(svc, export_svc, tmp_path)
        reader = csv.reader(
            io.StringIO(out.read_text(encoding="utf-8-sig")), delimiter="\t"
        )
        rows = list(reader)
        # Find the row for 食べた (newest first in repo)
        tabeta_row = next(r for r in rows if r[0] == "食べた")
        assert tabeta_row[4] == "昨日ラーメンを食べた。"
        assert tabeta_row[5] == "昨天吃了拉麵。"

    def test_html_special_chars_escaped(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        svc.extract_and_save_word(
            "テスト",
            source_sentence="<b>テスト</b>文。",
        )
        out = tmp_path / "html_escape.tsv"
        export_svc.export_to_anki(out)
        content = out.read_text(encoding="utf-8-sig")
        assert "&lt;b&gt;" in content
        assert "<b>" not in content

    def test_returns_row_count(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        svc.extract_and_save_word("猫")
        svc.extract_and_save_word("犬")
        n = export_svc.export_to_anki(tmp_path / "count.tsv")
        assert n == 2

    def test_utf8_bom_written(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        svc.extract_and_save_word("猫")
        out = tmp_path / "bom.tsv"
        export_svc.export_to_anki(out)
        raw = out.read_bytes()
        assert raw[:3] == b"\xef\xbb\xbf", "Expected UTF-8 BOM"

    def test_empty_vocab_writes_zero_rows(
        self, export_svc: ExportService, tmp_path: Path
    ) -> None:
        out = tmp_path / "empty.tsv"
        n = export_svc.export_to_anki(out)
        assert n == 0
        assert out.read_text(encoding="utf-8-sig") == ""

    def test_header_comment_when_requested(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        svc.extract_and_save_word("猫")
        out = tmp_path / "header.tsv"
        export_svc.export_to_anki(out, include_header=True)
        lines = out.read_text(encoding="utf-8-sig").split("\n")
        assert lines[0].startswith("#")

    def test_custom_delimiter(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        svc.extract_and_save_word("猫")
        out = tmp_path / "pipe.csv"
        export_svc.export_to_anki(out, delimiter="|")
        first_line = out.read_text(encoding="utf-8-sig").split("\n")[0]
        assert "|" in first_line


# ─────────────────────────────────────────────────────────────────────────────
# TestExportServiceJson
# ─────────────────────────────────────────────────────────────────────────────

class TestExportServiceJson:
    """JSON export structure validation."""

    def _export(
        self,
        svc: VocabularyService,
        export_svc: ExportService,
        tmp_path: Path,
    ) -> dict:
        svc.extract_and_save_word(
            "食べた",
            translation="ate",
            source_sentence="食べた例文。",
            target_sentence="範例。",
        )
        out = tmp_path / "backup.json"
        export_svc.export_to_json(out)
        return json.loads(out.read_text(encoding="utf-8"))

    def test_top_level_keys_present(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        data = self._export(svc, export_svc, tmp_path)
        assert "exported_at" in data
        assert "source_lang" in data
        assert "count" in data
        assert "entries" in data

    def test_count_matches_entries_length(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        data = self._export(svc, export_svc, tmp_path)
        assert data["count"] == len(data["entries"])

    def test_entry_fields_present(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        data = self._export(svc, export_svc, tmp_path)
        entry = data["entries"][0]
        for key in ("id", "surface_form", "base_form", "reading", "part_of_speech",
                    "translation", "notes", "review_count", "created_at", "examples"):
            assert key in entry, f"Missing key: {key}"

    def test_examples_nested_correctly(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        data = self._export(svc, export_svc, tmp_path)
        examples = data["entries"][0]["examples"]
        assert len(examples) == 1
        assert examples[0]["source_text"] == "食べた例文。"
        assert examples[0]["target_text"] == "範例。"

    def test_returns_entry_count(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        svc.extract_and_save_word("猫")
        svc.extract_and_save_word("犬")
        out = tmp_path / "two.json"
        n = export_svc.export_to_json(out)
        assert n == 2

    def test_japanese_unicode_preserved(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        svc.extract_and_save_word("食べる")
        out = tmp_path / "unicode.json"
        export_svc.export_to_json(out)
        raw = out.read_text(encoding="utf-8")
        assert "食べる" in raw  # ensure_ascii=False


# ─────────────────────────────────────────────────────────────────────────────
# TestExportServiceCsv
# ─────────────────────────────────────────────────────────────────────────────

class TestExportServiceCsv:
    """CSV export format validation."""

    def test_header_row_written(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        svc.extract_and_save_word("猫")
        out = tmp_path / "vocab.csv"
        export_svc.export_to_csv(out)
        lines = out.read_text(encoding="utf-8-sig").strip().split("\n")
        # First row is header
        assert "surface" in lines[0]

    def test_data_row_count(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        svc.extract_and_save_word("猫")
        svc.extract_and_save_word("犬")
        out = tmp_path / "two.csv"
        n = export_svc.export_to_csv(out)
        assert n == 2
        reader = csv.DictReader(
            io.StringIO(out.read_text(encoding="utf-8-sig"))
        )
        rows = list(reader)
        assert len(rows) == 2

    def test_comma_delimiter_default(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        svc.extract_and_save_word("猫")
        out = tmp_path / "comma.csv"
        export_svc.export_to_csv(out)
        first_line = out.read_text(encoding="utf-8-sig").split("\n")[0]
        assert "," in first_line


# ─────────────────────────────────────────────────────────────────────────────
# TestExportServiceDispatch
# ─────────────────────────────────────────────────────────────────────────────

class TestExportServiceDispatch:
    """export() format dispatch."""

    def test_dispatch_anki(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        svc.extract_and_save_word("猫")
        n = export_svc.export("anki", tmp_path / "d.tsv")
        assert n == 1

    def test_dispatch_json(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        svc.extract_and_save_word("猫")
        n = export_svc.export("json", tmp_path / "d.json")
        assert n == 1

    def test_dispatch_csv(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        svc.extract_and_save_word("猫")
        n = export_svc.export("csv", tmp_path / "d.csv")
        assert n == 1

    def test_dispatch_case_insensitive(
        self, svc: VocabularyService, export_svc: ExportService, tmp_path: Path
    ) -> None:
        svc.extract_and_save_word("猫")
        n = export_svc.export("ANKI", tmp_path / "upper.tsv")
        assert n == 1

    def test_unknown_format_raises(
        self, export_svc: ExportService, tmp_path: Path
    ) -> None:
        with pytest.raises(UnsupportedExportFormatError):
            export_svc.export("docx", tmp_path / "bad.docx")
