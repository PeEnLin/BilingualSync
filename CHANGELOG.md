# Changelog

All notable changes to **BilingualSync** are documented here.

## [Unreleased]

### Added — ASR + Translation Pipeline (Phase 5 refactor)

#### `core/asr/transcriber.py`
- **`cpu_threads=4`** constructor parameter passed to CTranslate2 to cap thread
  usage and prevent thermal throttling on Apple Silicon (M-series).
- **3-tier `get_model` fallback chain**:
  1. `local_files_only=True` — instant cold-start, no Hugging Face Hub round-trip
  2. Standard network load — first-run automatic download
  3. `device="cpu"`, `compute_type="float32"` — compatibility last resort
- **`_is_translatable` helper** — pure punctuation / symbol segments skip
  translation rather than consuming API quota.
- **`initial_prompt`** (`"こんにちは。日本語の字幕です。"`) to steer Whisper toward
  Japanese punctuation and oral sentence patterns.
- **`condition_on_previous_text=True`** for better inter-sentence coherence.
- **Optimised VAD parameters** — `min_silence_duration_ms=700`,
  `speech_pad_ms=300` — preventing sokuon (っ) and long vowels from being
  clipped at segment boundaries.
- **`max_retries=3`** in `translate_text` (was 2).
- **Graceful final fallback** — on total provider failure the original Japanese
  text is returned (not `""`) so the Chinese column always has visible content.

#### `app.py`
- `_init_transcriber` passes `cpu_threads=4` to `AudioTranscriber`.
- Removed hard-coded `beam_size=5` override from sidebar ASR handler —
  optimised defaults (`beam_size=3`) from the transcriber are used throughout.
- `_init_transcriber` docstring explains singleton semantics and `local_files_only`.

### Changed — Earlier sessions

#### UI / CSS
- File-uploader contrast: explicit colour overrides for `[data-testid="stFileUploader"]`
  and child elements for legibility on dark sidebar.
- Dynamic-width Japanese token chips via `display:inline-flex` / `width:fit-content`.
- Chip toggle to `type="primary"` + `✓ ` prefix on word-save for instant feedback.

#### SRT parsing & alignment
- `_parse_and_translate_ja`: single-SRT upload path auto-translates pure
  Japanese SRT to Traditional Chinese.
- "🌐 自動翻譯繁中 (純日文 SRT)" sidebar button wired to `_parse_and_translate_ja`.

### Fixed
- `st.rerun()` in word-click handler for immediate chip state refresh.
- `AlignedPair.target.text` guaranteed non-empty — falls back to source text.

### Tests
- `test_transcriber.py`: covers `cpu_threads`, 3-tier fallback, updated
  `vad_parameters` / `initial_prompt` assertion, and
  `test_translate_text_all_fail_preserves_source`.
- `test_ui_app.py` (new): AppTest integration for pure-JA SRT and chip toggle.
- Total: **182 passing** (up from 175).
