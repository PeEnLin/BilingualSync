"""
app.py
~~~~~~
BilingualSync – Streamlit Bilingual Subtitle Learning Assistant

Run with:

    streamlit run app.py

Feature Structure
-----------------
Sidebar
  ├─ Language Selector (繁體中文 / English)
  ├─ Video ASR & Translation (Whisper + Fallback Chain)
  ├─ Subtitle Upload (Japanese / Chinese .srt)
  └─ Vocabulary Vault
       ├─ Word Count
       ├─ Search
       ├─ Vocabulary Cards
       └─ Export Anki TSV

Main Area
  ├─ Empty State (before upload)
  └─ Bilingual Subtitle Reader
       ├─ Left Column: Japanese Sentences (Token Chips, click to add to vocabulary)
       └─ Right Column: Chinese Sentences
"""

from __future__ import annotations

import html
import logging
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# i18n Translation Dictionary
# ─────────────────────────────────────────────────────────────────────────────
TRANSLATIONS: dict[str, dict[str, str]] = {
    "繁體中文": {
        "page_title": "BilingualSync - 雙語字幕對照閱讀器",
        "title": "🎬 雙語字幕對照閱讀器",
        "sidebar_lang_label": "🌐 介面語言 / Language",
        "sidebar_header": "影片處理與控制台",
        "upload_label": "上傳影音檔案 (MP4, MKV, WebM, MP3, WAV)",
        "upload_help": "請上傳欲進行字幕生成與單字分析的影音檔",
        "legend_tip": "點擊日文詞彙即可加入單字庫・顏色代表詞性：",
        "pos_verb": "動詞",
        "pos_noun": "名詞",
        "pos_adj": "形容詞",
        "pos_na_adj": "形容動詞",
        "pos_adv": "副詞",
        "extracting_audio": "正在抽取音訊 (FFmpeg)...",
        "transcribing": "正在進行語音轉錄 (Whisper ASR)...",
        "transcribe_complete": "轉錄完成！共 {count} 句字幕",
        "translating": "正在進行中文語意對齊翻譯...",
        "unit_segments": "句",
        "btn_export_srt": "匯出雙語 SRT",
        "btn_export_vocab": "匯出單字本",
        "no_data_hint": "請先上傳影片以開始辨識與學習",
        "about_app": "# BilingualSync\n雙語字幕驅動的日文單字學習系統",
        "brand_subtitle": "BILINGUAL SUBTITLE LEARNING",
        "video_asr_caption": "透過離線語音聽打與機器翻譯，直接產出雙語字幕",
        "model_select_label": "Whisper 模型精度",
        "model_opt_high": "高精確度",
        "model_opt_fast": "較快",
        "model_select_help": "medium 模型具備最高日語識別率 (建議)；small 模型運算較快",
        "btn_start_transcribe": "🚀 開始深度辨識與翻譯",
        "warn_no_speech": "語音中未識別出有效語句，請確認音訊是否包含清楚日語。",
        "err_transcribe": "辨識處理失敗",
        "srt_section_header": "📂 既有雙語 SRT 字幕上傳",
        "ja_sub_label": "🇯🇵 日文字幕 (.srt)",
        "ja_sub_help": "上傳日文原版字幕檔（.srt 格式）",
        "zh_sub_label": "🇹🇼 中文字幕 (.srt)",
        "zh_sub_help": "上傳中文翻譯字幕檔（.srt 格式）",
        "btn_parse_align": "🔍 解析並對齊字幕",
        "parse_spinner": "解析字幕中，請稍候…",
        "parse_error_msg": "解析失敗",
        "parse_success_msg": "成功對齊 {count} 個字幕句對！",
        "btn_auto_translate": "🌐 自動翻譯繁中 (純日文 SRT)",
        "auto_translate_help": "若僅有日文字幕檔，一鍵自動逐句翻譯為繁體中文並對齊雙語字幕",
        "auto_translate_spinner": "🌐 正在自動翻譯繁體中文，請稍候…",
        "auto_translate_error_msg": "翻譯失敗",
        "auto_translate_success_msg": "成功翻譯並生成 {count} 個繁中對應字幕句對！",
        "empty_ja_srt_error": "日文字幕檔案內容為空或無法解析有效的字幕區塊。",
        "vocab_vault_header": "📚 單字庫面板 (Vocabulary Vault)",
        "unit_words": "個單字",
        "vocab_search_placeholder": "輸入詞彙、原形或翻譯…",
        "vocab_empty_hint": "尚未收錄任何單字。<br>點擊日文字幕中的詞彙即可加入！",
        "vocab_base_form": "原型：",
        "vocab_reading": "讀音（片假名）：",
        "vocab_context_sentence": "所屬例句",
        "vocab_delete_tooltip": "刪除「{word}」",
        "vocab_deleted_toast": "已從單字庫刪除：{word}",
        "vocab_showing_count": "顯示前 {shown} 筆，共 {total} 筆結果",
        "btn_clear_all_vocab": "🗑️ 清空所有單字",
        "clear_all_vocab_help": "清空單字庫內所有單字",
        "clear_all_vocab_warning": "⚠️ 確定要清空所有單字嗎？此動作無法復原！",
        "btn_confirm_clear": "⚠️ 確認清空",
        "btn_cancel": "取消",
        "vocab_cleared_toast": "已清空所有單字庫資料！",
        "btn_export_anki": "⬇️  匯出 Anki (.tsv)",
        "export_anki_help": "下載 Anki 匯入格式（Tab 分隔，UTF-8 BOM）",
        "export_anki_disabled_help": "尚無單字可匯出",
        "export_anki_tip": "💡 提示：匯出的 .tsv 檔符合 Anki 欄位標準，開啟 Anki 點選「匯入檔案」即可直接生成單字牌組。",
        "empty_welcome_title": "歡迎使用 BilingualSync",
        "empty_welcome_desc": "請在左側側邊欄上傳<strong>日文</strong>與<strong>中文</strong>字幕檔，<br>或直接上傳影片檔案進行語音辨識與翻譯，即可開始互動學習。",
        "feature1_title": "智慧字幕對齊",
        "feature1_desc": "滑動視窗時間戳重疊演算法<br>自動匹配雙語句對",
        "feature2_title": "一鍵加入單字庫",
        "feature2_desc": "點擊任意日文詞彙<br>自動解析原形、讀音與詞性",
        "feature3_title": "Anki 匯出",
        "feature3_desc": "完整例句上下文<br>一鍵匯出 TSV 單字卡",
        "label_base": "原形",
        "no_matching_zh": "（無對應中文字幕）",
        "toast_added_vocab": "加入單字庫：**{word}**",
        "toast_save_failed": "儲存失敗：{err}",
    },
    "English": {
        "page_title": "BilingualSync - Bilingual Subtitle Reader",
        "title": "🎬 Bilingual Subtitle Reader",
        "sidebar_lang_label": "🌐 Interface Language / 語言",
        "sidebar_header": "Video Processing & Control",
        "upload_label": "Upload Media File (MP4, MKV, WebM, MP3, WAV)",
        "upload_help": "Upload a media file for subtitle transcription and token analysis",
        "legend_tip": "Click any Japanese token to save to vocabulary. Colors indicate POS:",
        "pos_verb": "Verb",
        "pos_noun": "Noun",
        "pos_adj": "Adj",
        "pos_na_adj": "Na-Adj",
        "pos_adv": "Adv",
        "extracting_audio": "Extracting audio (FFmpeg)...",
        "transcribing": "Transcribing audio with Whisper ASR...",
        "transcribe_complete": "Transcription complete: {count} segments",
        "translating": "Aligning and translating sentences...",
        "unit_segments": "segments",
        "btn_export_srt": "Export Bilingual SRT",
        "btn_export_vocab": "Export Vocab List",
        "no_data_hint": "Please upload a video file to start transcription",
        "about_app": "# BilingualSync\nBilingual Subtitle Japanese Vocabulary Learning Assistant",
        "brand_subtitle": "BILINGUAL SUBTITLE LEARNING",
        "video_asr_caption": "Offline speech-to-text & machine translation for bilingual subtitles",
        "model_select_label": "Whisper Model Size",
        "model_opt_high": "High accuracy",
        "model_opt_fast": "Faster",
        "model_select_help": "medium offers the highest Japanese accuracy (recommended); small is faster",
        "btn_start_transcribe": "🚀 Start Transcription & Translation",
        "warn_no_speech": "No valid speech detected in the audio. Please ensure clear Japanese audio.",
        "err_transcribe": "Transcription failed",
        "srt_section_header": "📂 Upload Subtitle Files (.srt)",
        "ja_sub_label": "🇯🇵 Japanese Subtitles (.srt)",
        "ja_sub_help": "Upload original Japanese subtitle file (.srt format)",
        "zh_sub_label": "🇹🇼 Chinese Subtitles (.srt)",
        "zh_sub_help": "Upload Chinese translated subtitle file (.srt format)",
        "btn_parse_align": "🔍 Parse & Align Subtitles",
        "parse_spinner": "Parsing subtitles, please wait…",
        "parse_error_msg": "Parsing failed",
        "parse_success_msg": "Successfully aligned {count} subtitle segments!",
        "btn_auto_translate": "🌐 Auto-Translate to Chinese",
        "auto_translate_help": "Automatically translate Japanese subtitles to Traditional Chinese and align pairs",
        "auto_translate_spinner": "🌐 Auto-translating to Chinese, please wait…",
        "auto_translate_error_msg": "Translation failed",
        "auto_translate_success_msg": "Successfully translated and generated {count} subtitle segments!",
        "empty_ja_srt_error": "Japanese subtitle file is empty or contains no valid subtitle blocks.",
        "vocab_vault_header": "📚 Vocabulary Vault",
        "unit_words": "words",
        "vocab_search_placeholder": "Search surface, base form, or meaning…",
        "vocab_empty_hint": "No vocabulary saved yet.<br>Click any token in the subtitles to add!",
        "vocab_base_form": "Base Form:",
        "vocab_reading": "Reading:",
        "vocab_context_sentence": "Context Sentence",
        "vocab_delete_tooltip": "Delete '{word}'",
        "vocab_deleted_toast": "Deleted from vocabulary: {word}",
        "vocab_showing_count": "Showing first {shown} of {total} results",
        "btn_clear_all_vocab": "🗑️ Clear All Vocabulary",
        "clear_all_vocab_help": "Clear all words in Vocabulary Vault",
        "clear_all_vocab_warning": "⚠️ Are you sure you want to clear all vocabulary? This action cannot be undone!",
        "btn_confirm_clear": "⚠️ Confirm Clear",
        "btn_cancel": "Cancel",
        "vocab_cleared_toast": "All vocabulary cleared!",
        "btn_export_anki": "⬇️  Export Anki (.tsv)",
        "export_anki_help": "Download Anki import format (Tab-separated, UTF-8 BOM)",
        "export_anki_disabled_help": "No vocabulary to export",
        "export_anki_tip": "💡 Tip: The exported .tsv follows standard Anki format. Open Anki and click 'Import File' to generate your flashcard deck.",
        "empty_welcome_title": "Welcome to BilingualSync",
        "empty_welcome_desc": "Upload <strong>Japanese</strong> and <strong>Chinese</strong> subtitle files from the sidebar,<br>or transcribe directly from a video file to begin interactive learning.",
        "feature1_title": "Smart Alignment",
        "feature1_desc": "Sliding-window timestamp overlap<br>automatically matches bilingual pairs",
        "feature2_title": "One-Click Vocabulary",
        "feature2_desc": "Click any Japanese token<br>to parse base form, reading & POS",
        "feature3_title": "Anki Export",
        "feature3_desc": "Full sentence context<br>one-click TSV flashcard export",
        "label_base": "Base",
        "no_matching_zh": "(No matching Chinese subtitle)",
        "toast_added_vocab": "Added to vocabulary: **{word}**",
        "toast_save_failed": "Failed to save: {err}",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Page config – MUST be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
_curr_lang = st.session_state.get("ui_language", "繁體中文")
_t_init = TRANSLATIONS.get(_curr_lang, TRANSLATIONS["繁體中文"])

st.set_page_config(
    page_title=_t_init["page_title"],
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": _t_init["about_app"],
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS injection
# ─────────────────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached resource initialisation
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⚙️  Connecting to database…")
def _init_db_connection():
    """
    Open a long-lived SQLite connection for the Streamlit process.

    Returns a raw ``sqlite3.Connection`` kept alive for the app lifetime.
    ``check_same_thread=False`` is required for Streamlit's threading model.
    """
    from database.connection import DatabaseConnection
    from config.settings import get_settings

    settings = get_settings()
    db_ctx = DatabaseConnection(
        db_path=settings.db_path,
        check_same_thread=False,
    )
    conn = db_ctx.__enter__()  # We intentionally keep this open for app lifetime
    return conn


@st.cache_resource(show_spinner="🔬 Loading Japanese morphological analyzer…")
def _init_tokenizer():
    """Initialise Janome tokeniser once per process (heavy model load)."""
    from core.nlp.japanese_engine import JapaneseTokenizer
    return JapaneseTokenizer()


def _get_repo():
    from database.repository import VocabularyRepository
    return VocabularyRepository(_init_db_connection())


def _get_vocab_svc():
    from services.vocab_service import VocabularyService
    return VocabularyService(repo=_get_repo(), tokenizer=_init_tokenizer())


def _get_export_svc():
    from services.export_service import ExportService
    return ExportService(repo=_get_repo())


@st.cache_resource(show_spinner="🎙️ Loading speech recognition & translation engine…")
def _init_transcriber():
    """Instantiate AudioTranscriber once per Streamlit process lifetime.

    ``@st.cache_resource`` guarantees a single global instance – the Whisper
    model is loaded exactly once and reused across all re-runs within the same
    server process.
    """
    from core.asr.transcriber import AudioTranscriber

    return AudioTranscriber(
        default_model_size="medium",
        device="auto",
        compute_type="int8",
        cpu_threads=4,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_session() -> None:
    defaults = {
        "aligned_pairs": [],          # list[AlignedPair]
        "saved_surfaces": set(),      # surfaces already in DB this session
        "parse_error": None,          # str | None
        "vocab_search": "",           # sidebar search query
        "confirm_clear_vocab": False, # confirm state for clearing all words
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────────────────
# SRT parsing helper
# ─────────────────────────────────────────────────────────────────────────────

def _parse_and_align(ja_bytes: bytes, zh_bytes: bytes) -> None:
    """Parse two SRT byte streams and store aligned pairs in session state."""
    from core.aligner.srt_parser import SRTParser
    from core.aligner.text_aligner import TextAligner
    from core.exceptions import SRTParseError

    st.session_state.parse_error = None
    parser = SRTParser()

    try:
        # Write to temp files so parse_file() handles encoding detection
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as ja_f:
            ja_f.write(ja_bytes)
            ja_path = ja_f.name
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as zh_f:
            zh_f.write(zh_bytes)
            zh_path = zh_f.name

        ja_blocks = parser.parse_file(ja_path)
        zh_blocks = parser.parse_file(zh_path)
    except SRTParseError as exc:
        st.session_state.parse_error = str(exc)
        st.session_state.aligned_pairs = []
        return

    aligner = TextAligner(window_ms=5_000, min_overlap_ratio=0.05)
    pairs = aligner.align(ja_blocks, zh_blocks)
    st.session_state.aligned_pairs = pairs

    # Sync saved surfaces from DB
    repo = _get_repo()
    all_entries = repo.list_all(source_lang="ja", limit=10_000)
    st.session_state.saved_surfaces = {e.surface_form for e in all_entries}


def _parse_and_translate_ja(ja_bytes: bytes, t: dict[str, str] | None = None) -> None:
    """Parse pure Japanese SRT, automatically translate to Traditional Chinese, and store aligned pairs."""
    from core.aligner.srt_parser import SRTParser
    from core.base import AlignedPair
    from core.exceptions import SRTParseError

    if t is None:
        t = TRANSLATIONS.get(st.session_state.get("ui_language", "繁體中文"), TRANSLATIONS["繁體中文"])

    st.session_state.parse_error = None
    parser = SRTParser()

    try:
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as ja_f:
            ja_f.write(ja_bytes)
            ja_path = ja_f.name
        ja_blocks = parser.parse_file(ja_path)
    except SRTParseError as exc:
        st.session_state.parse_error = str(exc)
        st.session_state.aligned_pairs = []
        return
    finally:
        try:
            Path(ja_path).unlink(missing_ok=True)
        except Exception:
            pass

    if not ja_blocks:
        st.session_state.parse_error = t["empty_ja_srt_error"]
        st.session_state.aligned_pairs = []
        return

    transcriber = _init_transcriber()
    pairs = transcriber.translate_subtitle_blocks(ja_blocks, source="ja", target="zh-TW")
    aligned = [
        AlignedPair(source=ja_b, target=zh_b, overlap_ratio=1.0)
        for ja_b, zh_b in pairs
    ]
    st.session_state.aligned_pairs = aligned

    # Sync saved surfaces from DB
    repo = _get_repo()
    all_entries = repo.list_all(source_lang="ja", limit=10_000)
    st.session_state.saved_surfaces = {e.surface_form for e in all_entries}


# ─────────────────────────────────────────────────────────────────────────────
# Anki export helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_anki_bytes() -> bytes:
    """Generate Anki TSV bytes in-memory for st.download_button."""
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
        tmp_path = tmp.name
    _get_export_svc().export_to_anki(tmp_path)
    return Path(tmp_path).read_bytes()


def _format_srt_timestamp(ms: int) -> str:
    """Format millisecond integer into standard SRT timestamp HH:MM:SS,mmm."""
    hours = ms // 3600000
    rem = ms % 3600000
    mins = rem // 60000
    rem %= 60000
    secs = rem // 1000
    millis = rem % 1000
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def _build_bilingual_srt_bytes(pairs: Sequence[Any]) -> bytes:
    """Serialize aligned bilingual pairs into UTF-8 encoded SRT content."""
    lines: list[str] = []
    for idx, pair in enumerate(pairs, start=1):
        src = pair.source
        tgt = pair.target
        start_ts = _format_srt_timestamp(src.start_ms)
        end_ts = _format_srt_timestamp(src.end_ms)
        lines.append(str(idx))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(src.text)
        if tgt and tgt.text.strip():
            lines.append(tgt.text.strip())
        lines.append("")
    return "\n".join(lines).encode("utf-8")



# ─────────────────────────────────────────────────────────────────────────────
# Part-of-speech display tokens & filter rules
# ─────────────────────────────────────────────────────────────────────────────
_SKIP_BUTTON_POS = frozenset({"助詞", "助動詞", "記号", "BOS/EOS", "接頭詞", "接尾"})

# Backward-compatibility fallback POS style
_POS_STYLE: dict[str, tuple[str, str, str]] = {
    "動詞":     ("🔵", "#6C63FF", "Verb"),
    "名詞":     ("🟢", "#00CEC9", "Noun"),
    "形容詞":   ("🟡", "#FDCB6E", "Adj"),
    "形容動詞": ("🟠", "#E17055", "Na-Adj"),
    "副詞":     ("🔴", "#FF7675", "Adv"),
}

def _get_pos_style(t: dict[str, str]) -> dict[str, tuple[str, str, str]]:
    """Return POS mapping with localized English / Traditional Chinese label."""
    return {
        "動詞":     ("🔵", "#6C63FF", t["pos_verb"]),
        "名詞":     ("🟢", "#00CEC9", t["pos_noun"]),
        "形容詞":   ("🟡", "#FDCB6E", t["pos_adj"]),
        "形容動詞": ("🟠", "#E17055", t["pos_na_adj"]),
        "副詞":     ("🔴", "#FF7675", t["pos_adv"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar() -> dict[str, str]:
    with st.sidebar:
        # ── Language selector at the very top ─────────────────────────────
        curr_lang = st.session_state.get("ui_language", "繁體中文")
        lang_choice = st.selectbox(
            label="🌐 Language / 語言",
            options=["繁體中文", "English"],
            index=0 if curr_lang == "繁體中文" else 1,
            key="ui_language",
        )
        t = TRANSLATIONS[lang_choice]
        pos_style = _get_pos_style(t)

        st.divider()

        # ── Brand header ──────────────────────────────────────────────────
        st.markdown(
            f"""
            <div style="text-align:center; padding: 0.2rem 0 0.8rem;">
                <div style="font-size:2.2rem;">🎌</div>
                <div style="font-size:1.25rem; font-weight:700; color:#A29BFE;">BilingualSync</div>
                <div style="font-size:0.72rem; color:#5a6078; letter-spacing:1px;">{t["brand_subtitle"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Video ASR Section ─────────────────────────────────────────────
        with st.expander(t["sidebar_header"], expanded=True):
            st.caption(f"🎧 {t['video_asr_caption']}")
            media_file = st.file_uploader(
                t["upload_label"],
                type=["mp4", "mkv", "mov", "mp3", "wav"],
                key="media_uploader",
                help=t["upload_help"],
            )
            def _format_model_opt(opt: str) -> str:
                return f"{opt} ({t['model_opt_high'] if opt == 'medium' else t['model_opt_fast']})"

            model_opt = st.selectbox(
                t["model_select_label"],
                options=["medium", "small"],
                format_func=_format_model_opt,
                index=0,
                key="whisper_model_select",
                help=t["model_select_help"],
            )
            asr_model_size = model_opt

            transcribe_disabled = (media_file is None)
            if st.button(
                t["btn_start_transcribe"],
                disabled=transcribe_disabled,
                use_container_width=True,
                type="primary",
                key="btn_media_transcribe",
            ):
                suffix = f".{media_file.name.split('.')[-1]}"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_media:
                    tmp_media.write(media_file.read())
                    tmp_media_path = tmp_media.name

                try:
                    with st.status(t["transcribing"], expanded=True) as status_box:
                        def on_progress(msg: str):
                            status_box.write(msg)
                            status_box.update(label=msg)

                        status_box.write(t["extracting_audio"])
                        status_box.write(t["transcribing"])
                        transcriber = _init_transcriber()
                        pairs = transcriber.transcribe_and_translate(
                            media_path=tmp_media_path,
                            model_size=asr_model_size,
                            language="ja",
                            progress_callback=on_progress,
                        )

                        if not pairs:
                            status_box.update(label=t["warn_no_speech"], state="error")
                            st.warning(f"⚠️ {t['warn_no_speech']}")
                        else:
                            from core.base import AlignedPair
                            aligned = [
                                AlignedPair(source=ja_b, target=zh_b, overlap_ratio=1.0)
                                for ja_b, zh_b in pairs
                            ]
                            st.session_state.aligned_pairs = aligned

                            # Refresh saved surfaces from DB
                            repo = _get_repo()
                            all_entries = repo.list_all(source_lang="ja", limit=10_000)
                            st.session_state.saved_surfaces = {e.surface_form for e in all_entries}
                            st.session_state.parse_error = None
                            complete_msg = t["transcribe_complete"].format(count=len(aligned))
                            status_box.update(
                                label=complete_msg,
                                state="complete",
                                expanded=False,
                            )
                            st.success(f"✅ {complete_msg}")
                            st.rerun()
                except Exception as exc:
                    st.error(f"{t['err_transcribe']}: {exc}")
                    logger.exception("Failed during ASR transcription: %s", exc)
                finally:
                    try:
                        Path(tmp_media_path).unlink(missing_ok=True)
                    except Exception:
                        pass

        # ── SRT Upload Section ────────────────────────────────────────────
        with st.expander(t["srt_section_header"], expanded=False):
            ja_file = st.file_uploader(
                t["ja_sub_label"],
                type=["srt"],
                key="ja_uploader",
                help=t["ja_sub_help"],
            )
            zh_file = st.file_uploader(
                t["zh_sub_label"],
                type=["srt"],
                key="zh_uploader",
                help=t["zh_sub_help"],
            )

            parse_disabled = not (ja_file and zh_file)
            if st.button(
                t["btn_parse_align"],
                disabled=parse_disabled,
                use_container_width=True,
                type="secondary",
                key="btn_srt_parse",
            ):
                with st.spinner(t["parse_spinner"]):
                    _parse_and_align(ja_file.read(), zh_file.read())
                if st.session_state.parse_error:
                    st.error(f"{t['parse_error_msg']}: {st.session_state.parse_error}")
                else:
                    n = len(st.session_state.aligned_pairs)
                    st.success(f"✅ {t['parse_success_msg'].format(count=n)}")
                    st.rerun()

            translate_disabled = (ja_file is None)
            if st.button(
                t["btn_auto_translate"],
                disabled=translate_disabled,
                use_container_width=True,
                type="primary" if (ja_file and not zh_file) else "secondary",
                key="btn_ja_auto_translate",
                help=t["auto_translate_help"],
            ):
                with st.spinner(t["auto_translate_spinner"]):
                    _parse_and_translate_ja(ja_file.read(), t=t)
                if st.session_state.parse_error:
                    st.error(f"{t['auto_translate_error_msg']}: {st.session_state.parse_error}")
                else:
                    n = len(st.session_state.aligned_pairs)
                    st.success(f"✅ {t['auto_translate_success_msg'].format(count=n)}")
                    st.rerun()

        st.divider()

        # ── Vocabulary Vault ──────────────────────────────────────────────
        st.markdown(f'<div class="section-header">{t["vocab_vault_header"]}</div>', unsafe_allow_html=True)

        vocab_svc = _get_vocab_svc()
        word_count = vocab_svc.get_word_count()

        word_unit = "word" if (word_count == 1 and lang_choice == "English") else t["unit_words"]
        st.markdown(
            f'<div style="margin-bottom:0.8rem;">'
            f'<span class="stat-badge">📖 {word_count} {word_unit}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Search box
        search_q = st.text_input(
            "Search Vocabulary",
            placeholder=t["vocab_search_placeholder"],
            key="vocab_search_input",
            label_visibility="collapsed",
        )

        # Word list
        if word_count == 0:
            st.markdown(
                f'<div style="color:#5a6078; font-size:0.82rem; padding:0.5rem 0;">{t["vocab_empty_hint"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            results = vocab_svc.search(search_q) if search_q else vocab_svc.get_all_saved_words(limit=100)
            for entry in results[:50]:
                examples = vocab_svc.get_examples(entry.id)
                ex_src = examples[0].source_text if examples else ""
                ex_tgt = examples[0].target_text if examples else ""

                surf_esc = html.escape(entry.surface_form)
                base_esc = html.escape(entry.base_form) if entry.base_form else "–"
                reading_esc = html.escape(entry.reading) if entry.reading else "–"
                pos_info = pos_style.get(entry.part_of_speech)
                pos_display = pos_info[2] if pos_info else (entry.part_of_speech or "")
                pos_esc = html.escape(pos_display)

                ex_html = ""
                if ex_src:
                    ex_src_esc = html.escape(ex_src)
                    ex_tgt_html = f'<div class="vocab-ex-tgt">🇹🇼 {html.escape(ex_tgt)}</div>' if ex_tgt else ""
                    ex_html = (
                        f'<div class="vocab-card-section">'
                        f'<div class="vocab-card-label">{t["vocab_context_sentence"]}</div>'
                        f'<div class="vocab-ex-src">🇯🇵 {ex_src_esc}</div>'
                        f'{ex_tgt_html}'
                        f'</div>'
                    )

                pos_badge_html = f'<span class="vocab-pos-badge">{pos_esc}</span>' if pos_esc else ""

                with st.container(border=True):
                    c_left, c_del = st.columns([5, 1])
                    with c_left:
                        st.markdown(
                            f'<div style="display:flex; align-items:center; gap:8px;">'
                            f'<span class="vocab-card-title">{surf_esc}</span>'
                            f'{pos_badge_html}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    with c_del:
                        del_help = t["vocab_delete_tooltip"].format(word=entry.surface_form)
                        if st.button("✕", key=f"del_vocab_{entry.id}", help=del_help):
                            vocab_svc.delete_word(entry.id)
                            st.session_state.saved_surfaces.discard(entry.surface_form)
                            st.toast(f"🗑️ {t['vocab_deleted_toast'].format(word=entry.surface_form)}", icon="✅")
                            st.rerun()

                    st.markdown(
                        f'<div class="vocab-card-body">'
                        f'  <div class="vocab-meta-row">'
                        f'    <span class="vocab-card-label">{t["vocab_base_form"]}</span>'
                        f'    <span class="vocab-card-val">{base_esc}</span>'
                        f'  </div>'
                        f'  <div class="vocab-meta-row">'
                        f'    <span class="vocab-card-label">{t["vocab_reading"]}</span>'
                        f'    <span class="vocab-card-val reading">{reading_esc}</span>'
                        f'  </div>'
                        f'</div>'
                        f'{ex_html}',
                        unsafe_allow_html=True,
                    )

            if len(results) > 50:
                st.caption(t["vocab_showing_count"].format(shown=50, total=len(results)))

            # ── Clear All Words Button ─────────────────────────────────────────
            st.markdown('<div style="height:0.3rem;"></div>', unsafe_allow_html=True)
            if not st.session_state.get("confirm_clear_vocab", False):
                if st.button(t["btn_clear_all_vocab"], use_container_width=True, help=t["clear_all_vocab_help"]):
                    st.session_state.confirm_clear_vocab = True
                    st.rerun()
            else:
                st.warning(t["clear_all_vocab_warning"])
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button(t["btn_confirm_clear"], type="primary", use_container_width=True):
                        vocab_svc.delete_all_words()
                        st.session_state.saved_surfaces.clear()
                        st.session_state.confirm_clear_vocab = False
                        st.toast(f"🗑️ {t['vocab_cleared_toast']}", icon="✅")
                        st.rerun()
                with col_no:
                    if st.button(t["btn_cancel"], use_container_width=True):
                        st.session_state.confirm_clear_vocab = False
                        st.rerun()

        st.divider()

        # Export button
        if word_count > 0:
            anki_bytes = _build_anki_bytes()
            st.download_button(
                label=f"⬇️  {t['btn_export_vocab']} (.tsv)",
                data=anki_bytes,
                file_name="bilingual_sync_anki.tsv",
                mime="text/tab-separated-values",
                use_container_width=True,
                help=t["export_anki_help"],
                key="btn_download_anki_vocab",
            )
        else:
            st.button(
                f"⬇️  {t['btn_export_vocab']} (.tsv)",
                disabled=True,
                use_container_width=True,
                help=t["export_anki_disabled_help"],
                key="btn_download_anki_vocab_disabled",
            )

        st.markdown(
            f'<div style="font-size:0.75rem; color:#8892a0; margin-top:0.45rem; line-height:1.45;">'
            f'{t["export_anki_tip"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    return t


# ─────────────────────────────────────────────────────────────────────────────
# Main area: Empty State
# ─────────────────────────────────────────────────────────────────────────────

def _render_empty_state(t: dict[str, str]) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="icon">🎬</div>
            <h3>{t["empty_welcome_title"]}</h3>
            <p>{t["empty_welcome_desc"]}</p>
            <div style="font-size:0.85rem; color:#8892a0; margin-top:0.6rem;">{t["no_data_hint"]}</div>
        </div>

        <div style="display:flex; gap:1rem; margin-top:2rem; justify-content:center; flex-wrap:wrap;">
            <div class="glass-card" style="min-width:200px; text-align:center; flex:1;">
                <div style="font-size:1.8rem; margin-bottom:0.4rem;">🔍</div>
                <div style="font-weight:600; color:#A29BFE;">{t["feature1_title"]}</div>
                <div style="font-size:0.8rem; color:#5a6078; margin-top:0.3rem;">{t["feature1_desc"]}</div>
            </div>
            <div class="glass-card" style="min-width:200px; text-align:center; flex:1;">
                <div style="font-size:1.8rem; margin-bottom:0.4rem;">🎯</div>
                <div style="font-weight:600; color:#A29BFE;">{t["feature2_title"]}</div>
                <div style="font-size:0.8rem; color:#5a6078; margin-top:0.3rem;">{t["feature2_desc"]}</div>
            </div>
            <div class="glass-card" style="min-width:200px; text-align:center; flex:1;">
                <div style="font-size:1.8rem; margin-bottom:0.4rem;">📤</div>
                <div style="font-weight:600; color:#A29BFE;">{t["feature3_title"]}</div>
                <div style="font-size:0.8rem; color:#5a6078; margin-top:0.3rem;">{t["feature3_desc"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main area: Bilingual Reader
# ─────────────────────────────────────────────────────────────────────────────

def _render_reader(t: dict[str, str]) -> None:
    pairs = st.session_state.aligned_pairs
    tokenizer = _init_tokenizer()
    vocab_svc = _get_vocab_svc()
    pos_style = _get_pos_style(t)

    # Header
    n_pairs = len(pairs)
    n_matched = sum(1 for p in pairs if p.target is not None)
    col_h1, col_h2 = st.columns([2.8, 1.2])
    with col_h1:
        st.markdown(
            f'<div style="font-size:1.4rem; font-weight:700; color:#A29BFE; margin-bottom:0.3rem;">'
            f'{t["title"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_h2:
        st.markdown(
            f'<div style="text-align:right; padding-top:0.3rem;">'
            f'<span class="stat-badge">📝 {n_matched}/{n_pairs} {t["unit_segments"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Subtitle action toolbar (Legend & Export SRT)
    col_tb1, col_tb2 = st.columns([3, 1])
    with col_tb1:
        st.markdown(
            f'<div style="font-size:0.78rem; color:#5a6078; padding-top:0.3rem;">'
            f'💡 {t["legend_tip"]} '
            f'🔵{t["pos_verb"]} 🟢{t["pos_noun"]} 🟡{t["pos_adj"]} 🟠{t["pos_na_adj"]} 🔴{t["pos_adv"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_tb2:
        st.download_button(
            label=f"📥 {t['btn_export_srt']}",
            data=_build_bilingual_srt_bytes(pairs),
            file_name="bilingual_subtitles.srt",
            mime="text/plain",
            use_container_width=True,
            key="btn_download_bilingual_srt",
        )

    st.markdown('<div style="margin-bottom:0.8rem;"></div>', unsafe_allow_html=True)

    # Render each sentence pair
    for pair_idx, pair in enumerate(pairs, start=1):
        src = pair.source
        tgt = pair.target

        ts_start = f"{src.start_ms // 1000 // 60:02d}:{src.start_ms // 1000 % 60:02d}"
        ts_end   = f"{src.end_ms // 1000 // 60:02d}:{src.end_ms // 1000 % 60:02d}"

        with st.container():
            col_ja, col_zh = st.columns([1.1, 1.0], gap="medium")

            # ── Japanese column ────────────────────────────────────────────
            with col_ja:
                with st.container(border=True):
                    st.markdown(
                        f'<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.35rem;">'
                        f'<div style="display:flex; align-items:center; gap:8px;">'
                        f'<span class="subtitle-index">#{src.index}</span>'
                        f'<span class="ts-chip">{ts_start}→{ts_end}</span>'
                        f'</div>'
                        f'<span class="ts-chip" style="opacity:0.75; font-size:0.7rem;">{pair_idx}/{n_pairs} {t["unit_segments"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Tokenise and render word buttons
                    try:
                        tokens = tokenizer.tokenize(src.text)
                    except Exception:
                        tokens = []

                    if tokens:
                        key_prefix = f"tok_{src.index}"
                        btn_indices = [
                            i for i, tok in enumerate(tokens)
                            if tok.part_of_speech not in _SKIP_BUTTON_POS and len(tok.surface.strip()) > 0
                        ]

                        if btn_indices:
                            # Render token buttons in Streamlit columns.
                            # CSS flexbox overrides the equal-width column constraints,
                            # allowing chips to naturally flow and wrap based on word length.
                            chip_cols = st.columns(len(btn_indices))
                            for c_col, btn_idx in zip(chip_cols, btn_indices):
                                with c_col:
                                    tok = tokens[btn_idx]
                                    surf = tok.surface
                                    base = tok.base_form
                                    pos = tok.part_of_speech
                                    emoji, color, pos_label = pos_style.get(pos, ("⚪", "#8892a0", pos))
                                    is_saved = surf in st.session_state.saved_surfaces
                                    btn_label = f"{'✓ ' if is_saved else ''}{surf}"
                                    btn_key = f"{key_prefix}_{btn_idx}"
                                    btn_type = "primary" if is_saved else "secondary"
                                    if st.button(
                                        btn_label,
                                        key=btn_key,
                                        help=f"{emoji} {pos_label}  |  {t['label_base']}: {base}",
                                        type=btn_type,
                                        use_container_width=False,
                                    ):
                                        _on_word_click(
                                            surface=surf,
                                            src_sentence=src.text,
                                            tgt_sentence=tgt.text if tgt else "",
                                            src_start_ms=src.start_ms,
                                            src_end_ms=src.end_ms,
                                            vocab_svc=vocab_svc,
                                            t=t,
                                        )

                        # Render full sentence as readable text below buttons
                        st.markdown(
                            f'<div class="ja-sentence" style="font-size:0.9rem; color:#c8cdd8; margin-top:0.35rem;">{src.text}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="ja-sentence" style="font-size:0.9rem; color:#c8cdd8; margin-top:0.35rem;">{src.text}</div>',
                            unsafe_allow_html=True,
                        )

            # ── Chinese column ─────────────────────────────────────────────
            with col_zh:
                with st.container(border=True):
                    if tgt and tgt.text.strip():
                        ts_start_zh = f"{tgt.start_ms // 1000 // 60:02d}:{tgt.start_ms // 1000 % 60:02d}"
                        ts_end_zh   = f"{tgt.end_ms // 1000 // 60:02d}:{tgt.end_ms // 1000 % 60:02d}"
                        st.markdown(
                            f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:0.35rem;">'
                            f'<span class="subtitle-index">#{tgt.index}</span>'
                            f'<span class="ts-chip">{ts_start_zh}→{ts_end_zh}</span>'
                            f'</div>'
                            f'<div class="zh-text" style="margin-top:0.35rem;">{tgt.text}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:0.35rem; opacity:0.35;">'
                            f'<span class="subtitle-index">–</span>'
                            f'</div>'
                            f'<div class="zh-text" style="opacity:0.35; margin-top:0.35rem;">{t["no_matching_zh"]}</div>',
                            unsafe_allow_html=True,
                        )


# ─────────────────────────────────────────────────────────────────────────────
# Word-click handler
# ─────────────────────────────────────────────────────────────────────────────

def _on_word_click(
    surface: str,
    src_sentence: str,
    tgt_sentence: str,
    src_start_ms: int,
    src_end_ms: int,
    vocab_svc,
    t: dict[str, str],
) -> None:
    """Save clicked word to DB and show toast notification."""
    try:
        entry = vocab_svc.extract_and_save_word(
            clicked_text=surface,
            source_sentence=src_sentence,
            target_sentence=tgt_sentence,
            source_start_ms=src_start_ms,
            source_end_ms=src_end_ms,
        )
        st.session_state.saved_surfaces.add(surface)
        reading = f"【{entry.reading}】" if entry.reading else ""
        base    = entry.base_form if entry.base_form != surface else ""
        detail  = f"{reading} {base}".strip()
        toast_msg = t["toast_added_vocab"].format(word=surface)
        st.toast(f"✅ {toast_msg}{' → ' + detail if detail else ''}", icon="🎌")
        st.rerun()
    except Exception as exc:
        st.toast(f"⚠️ {t['toast_save_failed'].format(err=exc)}", icon="❌")


# ─────────────────────────────────────────────────────────────────────────────
# App entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _inject_css()
    _init_session()

    # Pre-warm cached resources quietly
    _init_db_connection()
    _init_tokenizer()
    _init_transcriber()

    # Sidebar (selects language and returns translation dictionary `t`)
    t = _render_sidebar()

    # Main content
    st.markdown(
        '<div style="height:0.2rem;"></div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.aligned_pairs:
        _render_empty_state(t)
    else:
        _render_reader(t)


if __name__ == "__main__":
    main()
