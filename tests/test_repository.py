"""
tests/test_repository.py
~~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for :class:`~database.repository.VocabularyRepository`.

These tests exercise the full database stack – connection bootstrap, schema
creation, and all CRUD operations – using an **in-memory** SQLite database so
no files are written to disk and tests are fully isolated.

Test categories
---------------
``TestVocabularyRepositoryCreate``
    Happy-path and error-path for :meth:`add` and :meth:`add_example`.
``TestVocabularyRepositoryRead``
    :meth:`get_by_id`, :meth:`find_by_surface`, :meth:`list_all`, :meth:`count`.
``TestVocabularyRepositoryUpdate``
    :meth:`update` and :meth:`increment_review_count`.
``TestVocabularyRepositoryDelete``
    :meth:`delete` including FK cascade verification.
``TestRepositoryEdgeCases``
    Boundary conditions: empty DB, pagination, multi-language partitioning.
"""

from __future__ import annotations

import pytest

from database.connection import DatabaseConnection
from database.models import ExampleSentence, VocabularyEntry
from database.repository import VocabularyRepository
from core.exceptions import (
    DuplicateRecordError,
    RecordNotFoundError,
)
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def repo():
    """
    Yield a :class:`VocabularyRepository` backed by an in-memory SQLite
    database.  The connection (and all data) is torn down after each test.
    """
    with DatabaseConnection(db_path=Path(":memory:"), check_same_thread=True) as conn:
        yield VocabularyRepository(conn)


def _make_entry(
    surface: str = "食べた",
    base: str = "食べる",
    reading: str = "たべた",
    pos: str = "動詞",
    translation: str = "ate",
    source_lang: str = "ja",
    target_lang: str = "zh-tw",
) -> VocabularyEntry:
    """Factory helper for constructing a transient VocabularyEntry."""
    return VocabularyEntry(
        surface_form=surface,
        base_form=base,
        reading=reading,
        part_of_speech=pos,
        translation=translation,
        source_lang=source_lang,
        target_lang=target_lang,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CREATE
# ─────────────────────────────────────────────────────────────────────────────

class TestVocabularyRepositoryCreate:
    """Tests for add() and add_example()."""

    def test_add_returns_entry_with_id(self, repo: VocabularyRepository) -> None:
        """A newly inserted entry must have an integer id."""
        entry = repo.add(_make_entry())
        assert entry.id is not None
        assert isinstance(entry.id, int)
        assert entry.id > 0

    def test_add_persists_all_fields(self, repo: VocabularyRepository) -> None:
        """All fields supplied to add() must round-trip through the database."""
        original = _make_entry(
            surface="走った",
            base="走る",
            reading="はしった",
            pos="動詞",
            translation="ran",
        )
        saved = repo.add(original)
        fetched = repo.get_by_id(saved.id)  # type: ignore[arg-type]

        assert fetched.surface_form == "走った"
        assert fetched.base_form == "走る"
        assert fetched.reading == "はしった"
        assert fetched.part_of_speech == "動詞"
        assert fetched.translation == "ran"

    def test_add_duplicate_raises(self, repo: VocabularyRepository) -> None:
        """Adding the same (surface, source_lang, target_lang) twice must fail."""
        repo.add(_make_entry())
        with pytest.raises(DuplicateRecordError) as exc_info:
            repo.add(_make_entry())  # identical surface / lang pair
        assert "VocabularyEntry" in str(exc_info.value)

    def test_add_example_returns_with_id(self, repo: VocabularyRepository) -> None:
        """add_example() must return an ExampleSentence with a populated id."""
        entry = repo.add(_make_entry())
        example = ExampleSentence(
            vocab_id=entry.id,  # type: ignore[arg-type]
            source_text="昨日の夜、ラーメンを食べた。",
            target_text="昨晚我吃了拉麵。",
        )
        saved = repo.add_example(example)
        assert saved.id is not None
        assert saved.id > 0

    def test_add_example_invalid_vocab_id_raises(
        self, repo: VocabularyRepository
    ) -> None:
        """Adding an example with a non-existent vocab_id must raise RecordNotFoundError."""
        example = ExampleSentence(vocab_id=9999, source_text="ダミー文。")
        with pytest.raises(RecordNotFoundError):
            repo.add_example(example)


# ─────────────────────────────────────────────────────────────────────────────
# READ
# ─────────────────────────────────────────────────────────────────────────────

class TestVocabularyRepositoryRead:
    """Tests for get_by_id(), find_by_surface(), list_all(), count()."""

    def test_get_by_id_returns_correct_entry(self, repo: VocabularyRepository) -> None:
        entry = repo.add(_make_entry(surface="飲む", base="飲む", translation="drink"))
        fetched = repo.get_by_id(entry.id)  # type: ignore[arg-type]
        assert fetched.id == entry.id
        assert fetched.surface_form == "飲む"

    def test_get_by_id_nonexistent_raises(self, repo: VocabularyRepository) -> None:
        with pytest.raises(RecordNotFoundError) as exc_info:
            repo.get_by_id(99999)
        assert "VocabularyEntry" in exc_info.value.message
        assert exc_info.value.context["identifier"] == 99999

    def test_find_by_surface_hit(self, repo: VocabularyRepository) -> None:
        repo.add(_make_entry(surface="見る", base="見る", translation="see"))
        result = repo.find_by_surface("見る", source_lang="ja", target_lang="zh-tw")
        assert result is not None
        assert result.surface_form == "見る"

    def test_find_by_surface_miss_returns_none(self, repo: VocabularyRepository) -> None:
        result = repo.find_by_surface("存在しない", source_lang="ja", target_lang="zh-tw")
        assert result is None

    def test_list_all_returns_all_entries(self, repo: VocabularyRepository) -> None:
        surfaces = ["猫", "犬", "魚"]
        for s in surfaces:
            repo.add(_make_entry(surface=s, base=s))
        entries = repo.list_all()
        assert len(entries) == 3

    def test_list_all_with_lang_filter(self, repo: VocabularyRepository) -> None:
        repo.add(_make_entry(surface="hello", base="hello", source_lang="en", target_lang="zh-tw"))
        repo.add(_make_entry(surface="食べた", base="食べる", source_lang="ja", target_lang="zh-tw"))
        ja_entries = repo.list_all(source_lang="ja")
        en_entries = repo.list_all(source_lang="en")
        assert len(ja_entries) == 1
        assert len(en_entries) == 1

    def test_list_all_pagination(self, repo: VocabularyRepository) -> None:
        surfaces = ["語" + str(i) for i in range(10)]
        for s in surfaces:
            repo.add(_make_entry(surface=s, base=s))
        page1 = repo.list_all(limit=4, offset=0)
        page2 = repo.list_all(limit=4, offset=4)
        page3 = repo.list_all(limit=4, offset=8)
        assert len(page1) == 4
        assert len(page2) == 4
        assert len(page3) == 2
        # No duplicates across pages
        all_ids = {e.id for e in page1 + page2 + page3}
        assert len(all_ids) == 10

    def test_count_empty(self, repo: VocabularyRepository) -> None:
        assert repo.count() == 0

    def test_count_after_inserts(self, repo: VocabularyRepository) -> None:
        repo.add(_make_entry(surface="A", base="A"))
        repo.add(_make_entry(surface="B", base="B"))
        assert repo.count() == 2

    def test_count_with_lang_filter(self, repo: VocabularyRepository) -> None:
        repo.add(_make_entry(surface="hello", base="hello", source_lang="en", target_lang="zh-tw"))
        repo.add(_make_entry(surface="食べた", base="食べる", source_lang="ja", target_lang="zh-tw"))
        assert repo.count(source_lang="en") == 1
        assert repo.count(source_lang="ja") == 1
        assert repo.count() == 2

    def test_get_examples_returns_associated_sentences(
        self, repo: VocabularyRepository
    ) -> None:
        entry = repo.add(_make_entry())
        repo.add_example(
            ExampleSentence(
                vocab_id=entry.id,  # type: ignore[arg-type]
                source_text="例文一。",
                target_text="範例一。",
            )
        )
        repo.add_example(
            ExampleSentence(
                vocab_id=entry.id,  # type: ignore[arg-type]
                source_text="例文二。",
                target_text="範例二。",
            )
        )
        examples = repo.get_examples(entry.id)  # type: ignore[arg-type]
        assert len(examples) == 2
        source_texts = {ex.source_text for ex in examples}
        assert "例文一。" in source_texts
        assert "例文二。" in source_texts


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE
# ─────────────────────────────────────────────────────────────────────────────

class TestVocabularyRepositoryUpdate:
    """Tests for update() and increment_review_count()."""

    def test_update_mutable_fields(self, repo: VocabularyRepository) -> None:
        entry = repo.add(_make_entry(translation=""))
        updated = repo.update(
            entry.model_copy(update={"translation": "吃了", "notes": "過去形"})
        )
        assert updated.translation == "吃了"
        assert updated.notes == "過去形"

    def test_update_refreshes_updated_at(self, repo: VocabularyRepository) -> None:
        entry = repo.add(_make_entry())
        updated = repo.update(entry.model_copy(update={"notes": "changed"}))
        assert updated.updated_at >= entry.updated_at

    def test_update_nonexistent_raises(self, repo: VocabularyRepository) -> None:
        phantom = _make_entry().model_copy(update={"id": 88888})
        with pytest.raises(RecordNotFoundError):
            repo.update(phantom)

    def test_update_without_id_raises(self, repo: VocabularyRepository) -> None:
        from core.exceptions import DatabaseError
        with pytest.raises(DatabaseError, match="id is None"):
            repo.update(_make_entry())

    def test_increment_review_count(self, repo: VocabularyRepository) -> None:
        entry = repo.add(_make_entry())
        assert entry.review_count == 0

        updated_once = repo.increment_review_count(entry.id)  # type: ignore[arg-type]
        assert updated_once.review_count == 1

        updated_twice = repo.increment_review_count(entry.id)  # type: ignore[arg-type]
        assert updated_twice.review_count == 2

    def test_increment_review_count_nonexistent_raises(
        self, repo: VocabularyRepository
    ) -> None:
        with pytest.raises(RecordNotFoundError):
            repo.increment_review_count(77777)


# ─────────────────────────────────────────────────────────────────────────────
# DELETE
# ─────────────────────────────────────────────────────────────────────────────

class TestVocabularyRepositoryDelete:
    """Tests for delete() including FK cascade."""

    def test_delete_removes_entry(self, repo: VocabularyRepository) -> None:
        entry = repo.add(_make_entry())
        repo.delete(entry.id)  # type: ignore[arg-type]
        with pytest.raises(RecordNotFoundError):
            repo.get_by_id(entry.id)  # type: ignore[arg-type]

    def test_delete_cascades_to_examples(self, repo: VocabularyRepository) -> None:
        """Deleting a VocabularyEntry must cascade-delete its ExampleSentences."""
        entry = repo.add(_make_entry())
        repo.add_example(
            ExampleSentence(
                vocab_id=entry.id,  # type: ignore[arg-type]
                source_text="消えるべき例文。",
            )
        )
        assert len(repo.get_examples(entry.id)) == 1  # type: ignore[arg-type]

        repo.delete(entry.id)  # type: ignore[arg-type]

        # After cascade delete, querying examples returns empty list
        # (get_examples does not raise for orphaned vocab_id,
        #  but there should be 0 rows)
        import sqlite3 as _sqlite3
        count_row = repo._conn.execute(
            "SELECT COUNT(*) FROM example_sentences WHERE vocab_id = ?",
            (entry.id,),
        ).fetchone()
        assert count_row[0] == 0

    def test_delete_nonexistent_raises(self, repo: VocabularyRepository) -> None:
        with pytest.raises(RecordNotFoundError):
            repo.delete(55555)

    def test_count_decreases_after_delete(self, repo: VocabularyRepository) -> None:
        e1 = repo.add(_make_entry(surface="A", base="A"))
        repo.add(_make_entry(surface="B", base="B"))
        assert repo.count() == 2
        repo.delete(e1.id)  # type: ignore[arg-type]
        assert repo.count() == 1


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestRepositoryEdgeCases:
    """Boundary and stress tests."""

    def test_same_surface_different_lang_pair_allowed(
        self, repo: VocabularyRepository
    ) -> None:
        """The same surface_form for different language pairs is NOT a duplicate."""
        repo.add(_make_entry(surface="open", base="open", source_lang="en", target_lang="zh-tw"))
        repo.add(_make_entry(surface="open", base="open", source_lang="en", target_lang="ja"))
        assert repo.count() == 2

    def test_whitespace_is_stripped_from_surface(
        self, repo: VocabularyRepository
    ) -> None:
        """Leading/trailing whitespace in surface_form must be stripped by the model."""
        entry = repo.add(_make_entry(surface="  猫  ", base="猫"))
        assert entry.surface_form == "猫"

    def test_multiple_examples_independent_entries(
        self, repo: VocabularyRepository
    ) -> None:
        entry = repo.add(_make_entry())
        for i in range(5):
            repo.add_example(
                ExampleSentence(
                    vocab_id=entry.id,  # type: ignore[arg-type]
                    source_text=f"例文{i}。",
                )
            )
        assert len(repo.get_examples(entry.id)) == 5  # type: ignore[arg-type]

    def test_delete_all_clears_records_and_examples(
        self, repo: VocabularyRepository
    ) -> None:
        e1 = repo.add(_make_entry(surface="食べる", base="食べる"))
        e2 = repo.add(_make_entry(surface="飲む", base="飲む"))
        repo.add_example(
            ExampleSentence(
                vocab_id=e1.id,  # type: ignore[arg-type]
                source_text="ご飯を食べる。",
            )
        )
        assert repo.count() == 2
        deleted = repo.delete_all()
        assert deleted == 2
        assert repo.count() == 0
        assert repo.list_all() == []

