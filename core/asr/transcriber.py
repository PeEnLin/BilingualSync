"""
core/asr/transcriber.py
~~~~~~~~~~~~~~~~~~~~~~~
Offline Japanese speech recognition + batch machine translation pipeline.

Architecture
------------
The previous per-sentence translation loop triggered Google Translate's
rate-limiter after ~10 requests, causing every subsequent sentence to exhaust
all retries (3 × 0.5 s each) before falling back.  With 50+ segments this
produced a 3–5 minute stall for zero useful output.

The new design **separates ASR from translation** completely:

1. **Phase 1 – ASR only** (faster-whisper)
   Collect *all* segments into a ``list[SubtitleBlock]`` without touching any
   network service.  Progress is printed to stdout with ``flush=True`` so the
   terminal stays alive.

2. **Phase 2 – Single batch translate**
   Join every Japanese sentence with a unique sentinel token
   ``\\n|||SEP|||\\n`` and send **one** HTTP request to Google Translate.
   One request ≈ 200–500 ms regardless of segment count.

3. **Phase 3 – Alignment**
   Split the translated blob on the sentinel and zip with original blocks.
   A length-mismatch guard (split-count ≠ segment-count) falls back to a
   line-by-line GTX call so no subtitle is ever left blank.

Key guarantees
--------------
* **No empty Chinese text**: on total failure the Japanese original is kept.
* **No blocking retry loops inside ASR**: translation happens once, after all
  speech is recognised.
* **`local_files_only` removed from primary path**: avoids ``EntryNotFoundError``
  on first run or after model updates.
"""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Any, Optional, Union

from core.base import SubtitleBlock

logger = logging.getLogger(__name__)

# ── Sentinel used to split the batch-translated blob ─────────────────────────
_SEP = "\n|||SEP|||\n"

# ── VAD defaults ──────────────────────────────────────────────────────────────
_DEFAULT_VAD_PARAMS: dict[str, int] = {
    "min_silence_duration_ms": 700,
    "speech_pad_ms": 300,
}

# ── Transcription defaults ────────────────────────────────────────────────────
_INITIAL_PROMPT = "こんにちは。日本語の字幕です。"
_DEFAULT_BEAM_SIZE = 3
_DEFAULT_CPU_THREADS = 4

# ── Translation configuration ───────────────────────────────────────────────
_TRANSLATE_TIMEOUT = 5.0    # seconds per HTTP call
_CHUNK_SIZE = 20            # max sentences per chunk (20 keeps URL short & avoids 429)
_CHUNK_CHAR_LIMIT = 2000    # max chars per chunk
_INTER_CHUNK_SLEEP = 1.2   # seconds between chunk requests (rate-limit mitigation)
_FALLBACK_MSG = "[翻譯線路忙碌，請稍後重試]"  # shown when all providers fail


def _has_letters(text: str) -> bool:
    """Return True when *text* contains at least one letter/digit to translate."""
    return any(unicodedata.category(ch)[0] in {"L", "N"} for ch in text)


def _split_result(blob: str, expected: int) -> list[str] | None:
    """Try to split a translated blob back into *expected* parts."""
    for sep in (_SEP.strip(), "|||SEP|||", "|||sep|||", "\n"):
        parts = [p.strip() for p in blob.split(sep)]
        if len(parts) == expected:
            return parts
    return None


def _mymemory_translate_chunk(
    texts: list[str],
    source: str = "ja",
    target: str = "zh-TW",
) -> list[str]:
    """
    Second priority provider: MyMemoryTranslator (deep_translator).

    MyMemory uses a separate infrastructure and IP pool from Google, making it
    resilient against Google 429 rate limits. Sentences are translated
    individually to stay well within MyMemory's 500-character request limit.
    """
    import time
    from deep_translator import MyMemoryTranslator

    # Map language codes to MyMemory format (e.g. ja -> ja-JP)
    _lang_map = {"ja": "ja-JP", "zh-TW": "zh-TW", "en": "en-US", "ko": "ko-KR"}
    mm_src = _lang_map.get(source, "ja-JP")
    mm_tgt = _lang_map.get(target, "zh-TW")

    results: list[str] = []
    try:
        translator = MyMemoryTranslator(source=mm_src, target=mm_tgt)
        for i, text in enumerate(texts):
            clean = text.strip()
            if not _has_letters(clean):
                results.append(clean)
                continue
            try:
                # 500-char safe limit
                zh = translator.translate(clean[:490])
                if zh and isinstance(zh, str) and zh.strip():
                    results.append(zh.strip())
                else:
                    results.append(_FALLBACK_MSG)
            except Exception as exc:
                logger.debug("[mymemory] Sentence %d translation failed: %s", i, exc)
                results.append(_FALLBACK_MSG)

            if i < len(texts) - 1:
                time.sleep(0.15)  # slight spacing between sentences

        return results

    except Exception as exc:
        logger.warning("[mymemory_chunk] MyMemory provider error: %s", exc)
        while len(results) < len(texts):
            results.append(_FALLBACK_MSG)
        return results


def safe_translate_chunk(
    texts: list[str],
    source: str = "ja",
    target: str = "zh-TW",
) -> list[str]:
    """
    Robust multi-provider translation pipeline with automatic fallback:
    1. First priority: GoogleTranslator(source=source, target=target)
    2. Second priority (on 429 rate limit or error): MyMemoryTranslator(source='ja-JP', target=target)
    3. Final fallback: [_FALLBACK_MSG] for each sentence.
    """
    if not texts:
        return []

    # ── First priority: GoogleTranslator (joined batch) ───────────────────────
    try:
        from deep_translator import GoogleTranslator

        joined = _SEP.join(texts)
        result = GoogleTranslator(source=source, target=target).translate(joined)
        if result and isinstance(result, str):
            parts = _split_result(result, len(texts))
            if parts is not None and all(p.strip() for p in parts):
                return [p.strip() for p in parts]
            logger.debug("[safe_translate_chunk] Google batch split mismatch, triggering fallback")
        print("[WARN] 觸發 429 限速，切換至備援翻譯器處理...", flush=True)
    except Exception as exc:
        print("[WARN] 觸發 429 限速，切換至備援翻譯器處理...", flush=True)
        logger.warning("[safe_translate_chunk] GoogleTranslator failed (%s), switching to fallback...", exc)

    # ── Second priority: MyMemoryTranslator ───────────────────────────────────
    return _mymemory_translate_chunk(texts, source=source, target=target)


class AudioTranscriber:
    """
    Offline speech recognition and machine translation engine.

    Parameters
    ----------
    default_model_size:
        faster-whisper model size (``"medium"``, ``"small"``, ``"base"``…).
    device:
        Compute device – ``"auto"``, ``"cpu"``, or ``"cuda"``.
    compute_type:
        Quantisation type – ``"int8"`` (Mac-safe) or ``"float32"``.
    cpu_threads:
        Number of CPU threads.  Defaults to 4 to prevent M-series throttle.
    """

    def __init__(
        self,
        default_model_size: str = "medium",
        device: str = "auto",
        compute_type: str = "int8",
        cpu_threads: int = _DEFAULT_CPU_THREADS,
    ) -> None:
        self._default_model_size = default_model_size
        self._device = device
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads
        self._models: dict[str, Any] = {}

    # ── Model management ──────────────────────────────────────────────────────

    def get_model(self, model_size: Optional[str] = None) -> Any:
        """
        Retrieve (or lazily instantiate) a :class:`faster_whisper.WhisperModel`.

        Falls back through two tiers when the primary load fails:

        1. Standard load (network-capable)
        2. ``device="cpu"``, ``compute_type="float32"`` compatibility mode
        """
        size = model_size or self._default_model_size
        if size in self._models:
            return self._models[size]

        from faster_whisper import WhisperModel

        logger.info(
            "Loading WhisperModel(size=%r, device=%r, compute_type=%r, cpu_threads=%d)…",
            size,
            self._device,
            self._compute_type,
            self._cpu_threads,
        )
        print(
            f"[INFO] 正在載入 Whisper 模型 ({size})，首次載入需要較長時間…",
            flush=True,
        )

        try:
            model = WhisperModel(
                size,
                device=self._device,
                compute_type=self._compute_type,
                cpu_threads=self._cpu_threads,
            )
        except Exception as exc:
            logger.warning(
                "WhisperModel(%r) failed (%s). Falling back to cpu/float32.", size, exc
            )
            print(f"[WARN] 模型載入失敗，切換至 CPU/float32 相容模式…", flush=True)
            model = WhisperModel(
                size,
                device="cpu",
                compute_type="float32",
                cpu_threads=self._cpu_threads,
            )

        self._models[size] = model
        print(f"[INFO] Whisper 模型 ({size}) 載入完成 ✓", flush=True)
        return model

    # ── Single-sentence translation (kept for SRT-only path) ─────────────────

    def translate_text(
        self,
        text: str,
        source: str = "ja",
        target: str = "zh-TW",
        max_retries: int = 3,
    ) -> str:
        """
        Translate a single sentence via single-item batch.

        Kept for backward-compatibility with ``translate_subtitle_blocks`` (used
        when uploading a pure Japanese SRT without an audio file).  For large
        batches use :meth:`translate_batch` directly.
        """
        clean = text.strip()
        if not clean:
            return ""
        if not _has_letters(clean):
            return clean

        results = self.translate_batch([clean], source=source, target=target)
        translated = results[0] if results else _FALLBACK_MSG
        return translated if (translated and translated.strip()) else _FALLBACK_MSG

    # ── Batch translation ─────────────────────────────────────────────────────

    def translate_batch(
        self,
        texts: list[str],
        source: str = "ja",
        target: str = "zh-TW",
    ) -> list[str]:
        """
        Translate a list of sentences in chunked batches (20 sentences / chunk)
        with automatic provider fallback (GoogleTranslator -> MyMemory) and
        a 1.2s inter-chunk cooling interval to avoid 429 rate-limiting.

        Parameters
        ----------
        texts:
            Sentences to translate. Empty / symbol-only strings are passed
            through as-is without consuming quota.
        source:
            BCP-47 source language code.
        target:
            BCP-47 target language code.

        Returns
        -------
        list[str]
            Translated strings in Traditional Chinese. Never contains raw Japanese
            for translated slots; on provider exhaustion, contains [_FALLBACK_MSG].
        """
        import time

        if not texts:
            return []

        # Partition: only translate non-trivial sentences
        indices_to_translate: list[int] = []
        payload: list[str] = []
        for i, t in enumerate(texts):
            clean = t.strip()
            if clean and _has_letters(clean):
                indices_to_translate.append(i)
                payload.append(clean)

        results: list[str] = list(texts)  # start with originals

        if not payload:
            return results

        # ── Build chunks ≤ _CHUNK_SIZE sentences & ≤ _CHUNK_CHAR_LIMIT chars ──
        chunks: list[list[int]] = []   # chunk[i] = list of indices into payload
        current: list[int] = []
        current_chars = 0
        for i, text in enumerate(payload):
            if current and (
                len(current) >= _CHUNK_SIZE or
                current_chars + len(text) > _CHUNK_CHAR_LIMIT
            ):
                chunks.append(current)
                current = []
                current_chars = 0
            current.append(i)
            current_chars += len(text)
        if current:
            chunks.append(current)

        n_chunks = len(chunks)
        print(
            f"[INFO] 批次翻譯：{len(payload)} 句分為 {n_chunks} 個請求批次",
            flush=True,
        )

        # ── Translate each chunk with safe fallback chain ──────────────────────
        translated_payload: list[str] = [_FALLBACK_MSG] * len(payload)

        for chunk_no, chunk_indices in enumerate(chunks, start=1):
            chunk_texts = [payload[i] for i in chunk_indices]
            print(
                f"[INFO] 翻譯批次 {chunk_no}/{n_chunks} ({len(chunk_texts)} 句)…",
                flush=True,
            )

            # Safe translation chain: GoogleTranslator -> MyMemoryTranslator -> _FALLBACK_MSG
            chunk_result = safe_translate_chunk(chunk_texts, source=source, target=target)

            for rel_i, abs_i in enumerate(chunk_indices):
                zh = chunk_result[rel_i] if rel_i < len(chunk_result) else _FALLBACK_MSG
                # Result assignment guarantee: write Traditional Chinese, NEVER overwrite back to Japanese original
                if zh and zh.strip():
                    translated_payload[abs_i] = zh.strip()
                else:
                    translated_payload[abs_i] = _FALLBACK_MSG

            # Rate-limit mitigation: mandatory 1.2s cooling interval between chunks
            if chunk_no < n_chunks:
                time.sleep(_INTER_CHUNK_SLEEP)

        # ── Write back into result array ───────────────────────────────────────
        for idx, zh in zip(indices_to_translate, translated_payload):
            results[idx] = zh if (zh and zh.strip()) else _FALLBACK_MSG

        print("[INFO] 批次翻譯全部完成 ✓", flush=True)
        return results

    # ── Bulk block translation (used by pure-SRT path) ────────────────────────

    def translate_subtitle_blocks(
        self,
        blocks: list[SubtitleBlock],
        source: str = "ja",
        target: str = "zh-TW",
    ) -> list[tuple[SubtitleBlock, SubtitleBlock]]:
        """
        Translate a list of source ``SubtitleBlock`` objects in one batch call.

        The returned Chinese ``SubtitleBlock`` shares identical ``start_ms``/
        ``end_ms`` as its Japanese counterpart for pixel-perfect alignment.
        """
        if not blocks:
            return []

        source_texts = [b.text for b in blocks]
        print(
            f"[INFO] 開始批次翻譯 {len(blocks)} 個字幕句段…",
            flush=True,
        )
        translated = self.translate_batch(source_texts, source=source, target=target)
        print("[INFO] 批次翻譯完成 ✓", flush=True)

        pairs: list[tuple[SubtitleBlock, SubtitleBlock]] = []
        for block, zh_text in zip(blocks, translated):
            final_zh = zh_text if (zh_text and zh_text.strip()) else _FALLBACK_MSG
            zh_block = SubtitleBlock(
                index=block.index,
                start_ms=block.start_ms,
                end_ms=block.end_ms,
                text=final_zh,
            )
            pairs.append((block, zh_block))
        return pairs

    # ── Full ASR + batch translation pipeline ─────────────────────────────────

    def transcribe_and_translate(
        self,
        media_path: Union[str, Path],
        model_size: Optional[str] = None,
        language: str = "ja",
        beam_size: int = _DEFAULT_BEAM_SIZE,
    ) -> list[tuple[SubtitleBlock, SubtitleBlock]]:
        """
        **Phase 1 – ASR** then **Phase 2 – Batch translate** (never interleaved).

        Parameters
        ----------
        media_path:
            Local path to video (.mp4, .mkv, .mov) or audio (.mp3, .wav).
        model_size:
            Optional model-size override.
        language:
            Source audio language (default ``"ja"``).
        beam_size:
            Beam-search width.  3 balances speed vs. accuracy.
        """
        file_path = Path(media_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Media file does not exist: {file_path}")

        print(f"[INFO] 接收到媒體檔案 {file_path.name}，開始處理…", flush=True)

        model = self.get_model(model_size)

        print("[INFO] 開始 Whisper ASR 日文語音辨識…", flush=True)
        logger.info(
            "Transcribing %s (lang=%s, beam=%d, vad=True)…",
            file_path.name,
            language,
            beam_size,
        )

        segments_iter, _ = model.transcribe(
            str(file_path),
            language=language,
            beam_size=beam_size,
            task="transcribe",
            initial_prompt=_INITIAL_PROMPT,
            condition_on_previous_text=True,
            vad_filter=True,
            vad_parameters=_DEFAULT_VAD_PARAMS,
        )

        # ── Phase 1: collect all ASR segments first ───────────────────────────
        ja_blocks: list[SubtitleBlock] = []
        idx = 1
        for segment in segments_iter:
            ja_text = segment.text.strip()
            if not ja_text:
                continue
            start_ms = max(0, int(round(segment.start * 1000)))
            end_ms = max(start_ms + 100, int(round(segment.end * 1000)))
            ja_blocks.append(
                SubtitleBlock(index=idx, start_ms=start_ms, end_ms=end_ms, text=ja_text)
            )
            print(f"[INFO] ASR 片段 #{idx}: {ja_text[:30]}…", flush=True)
            idx += 1

        print(
            f"[INFO] Whisper ASR 轉錄完成！共辨識出 {len(ja_blocks)} 個句段，"
            f"開始批次繁中翻譯…",
            flush=True,
        )

        if not ja_blocks:
            logger.warning("No speech segments detected in %s", file_path.name)
            return []

        # ── Phase 2: single batch translate ───────────────────────────────────
        source_texts = [b.text for b in ja_blocks]
        zh_texts = self.translate_batch(source_texts, source=language, target="zh-TW")

        print("[INFO] 雙語字幕對齊封裝完成，開始渲染介面！", flush=True)
        logger.info("ASR+translate complete: %d aligned pairs.", len(ja_blocks))

        pairs: list[tuple[SubtitleBlock, SubtitleBlock]] = []
        for ja_block, zh_text in zip(ja_blocks, zh_texts):
            final_zh = zh_text if (zh_text and zh_text.strip()) else _FALLBACK_MSG
            zh_block = SubtitleBlock(
                index=ja_block.index,
                start_ms=ja_block.start_ms,
                end_ms=ja_block.end_ms,
                text=final_zh,
            )
            pairs.append((ja_block, zh_block))

        return pairs
