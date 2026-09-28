"""
app.py
~~~~~~
BilingualSync – Streamlit 雙語字幕學習助理

啟動方式::

    streamlit run app.py

功能架構
--------
Sidebar
  ├─ 字幕上傳（日文 / 中文 .srt）
  ├─ 解析並對齊字幕
  └─ 單字庫面板（Vocabulary Vault）
       ├─ 單字總數
       ├─ 搜尋
       ├─ 單字清單（卡片式）
       └─ 匯出 Anki TSV

Main Area
  ├─ Empty State（未上傳時）
  └─ 雙語對照閱讀器
       ├─ 左欄：日文句（分詞 Tags，點擊加入單字庫）
       └─ 右欄：中文句
"""

from __future__ import annotations

import html
import logging
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config – MUST be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BilingualSync",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "# BilingualSync\n雙語字幕驅動的日文單字學習系統",
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

@st.cache_resource(show_spinner="⚙️  初始化資料庫連線…")
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


@st.cache_resource(show_spinner="🔬 載入日文形態素分析引擎…")
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


@st.cache_resource(show_spinner="🎙️ 初始化語音辨識與翻譯引擎…")
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


def _parse_and_translate_ja(ja_bytes: bytes) -> None:
    """Parse pure Japanese SRT, automatically translate to Traditional Chinese, and store aligned pairs."""
    from core.aligner.srt_parser import SRTParser
    from core.base import AlignedPair
    from core.exceptions import SRTParseError

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
        st.session_state.parse_error = "日文字幕檔案內容為空或無法解析有效的字幕區塊。"
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


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        # ── Brand header ──────────────────────────────────────────────────
        st.markdown(
            """
            <div style="text-align:center; padding: 0.5rem 0 1.2rem;">
                <div style="font-size:2.2rem;">🎌</div>
                <div style="font-size:1.25rem; font-weight:700; color:#A29BFE;">BilingualSync</div>
                <div style="font-size:0.72rem; color:#5a6078; letter-spacing:1px;">BILINGUAL SUBTITLE LEARNING</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Video ASR Section ─────────────────────────────────────────────
        with st.expander("🎬 影片自動生成雙語字幕", expanded=True):
            st.caption("🎧 透過離線語音聽打與機器翻譯，直接產出雙語字幕")
            media_file = st.file_uploader(
                "選擇影片/音訊檔",
                type=["mp4", "mkv", "mov", "mp3", "wav"],
                key="media_uploader",
                help="支援常見格式：.mp4, .mkv, .mov, .mp3, .wav",
            )
            model_opt = st.selectbox(
                "Whisper 模型精度",
                options=["medium (高精確度)", "small (較快)"],
                index=0,
                key="whisper_model_select",
                help="medium 模型具備最高日語識別率 (建議)；small 模型運算較快",
            )
            asr_model_size = "medium" if "medium" in model_opt else "small"

            transcribe_disabled = (media_file is None)
            if st.button(
                "🚀 開始深度辨識與翻譯",
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
                    with st.spinner("🎧 Phase 1：Whisper 日文 ASR 辨識中… Phase 2：批次繁中翻譯中（請查看終端機進度）"):
                        print("[INFO] 接收到上傳檔案，開始處理...", flush=True)
                        transcriber = _init_transcriber()
                        pairs = transcriber.transcribe_and_translate(
                            media_path=tmp_media_path,
                            model_size=asr_model_size,
                            language="ja",
                            # beam_size / VAD / initial_prompt use optimised defaults
                        )

                    if not pairs:
                        st.warning("⚠️ 語音中未識別出有效語句，請確認音訊是否包含清楚日語。")
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
                        st.success(f"✅ 成功辨識並生成 {len(aligned)} 個雙語句對！")
                        st.rerun()
                except Exception as exc:
                    st.error(f"辨識處理失敗：{exc}")
                    logger.exception("Failed during ASR transcription: %s", exc)
                finally:
                    try:
                        Path(tmp_media_path).unlink(missing_ok=True)
                    except Exception:
                        pass

        # ── SRT Upload Section ────────────────────────────────────────────
        with st.expander("📂 既有雙語 SRT 字幕上傳", expanded=False):
            ja_file = st.file_uploader(
                "🇯🇵 日文字幕 (.srt)",
                type=["srt"],
                key="ja_uploader",
                help="上傳日文原版字幕檔（.srt 格式）",
            )
            zh_file = st.file_uploader(
                "🇹🇼 中文字幕 (.srt)",
                type=["srt"],
                key="zh_uploader",
                help="上傳中文翻譯字幕檔（.srt 格式）",
            )

            parse_disabled = not (ja_file and zh_file)
            if st.button(
                "🔍 解析並對齊字幕",
                disabled=parse_disabled,
                use_container_width=True,
                type="secondary",
                key="btn_srt_parse",
            ):
                with st.spinner("解析字幕中，請稍候…"):
                    _parse_and_align(ja_file.read(), zh_file.read())
                if st.session_state.parse_error:
                    st.error(f"解析失敗：{st.session_state.parse_error}")
                else:
                    n = len(st.session_state.aligned_pairs)
                    st.success(f"✅ 成功對齊 {n} 個字幕句對！")
                    st.rerun()

            translate_disabled = (ja_file is None)
            if st.button(
                "🌐 自動翻譯繁中 (純日文 SRT)",
                disabled=translate_disabled,
                use_container_width=True,
                type="primary" if (ja_file and not zh_file) else "secondary",
                key="btn_ja_auto_translate",
                help="若僅有日文字幕檔，一鍵自動逐句翻譯為繁體中文並對齊雙語字幕",
            ):
                with st.spinner("🌐 正在自動翻譯繁體中文，請稍候…"):
                    _parse_and_translate_ja(ja_file.read())
                if st.session_state.parse_error:
                    st.error(f"翻譯失敗：{st.session_state.parse_error}")
                else:
                    n = len(st.session_state.aligned_pairs)
                    st.success(f"✅ 成功翻譯並生成 {n} 個繁中對應字幕句對！")
                    st.rerun()

        st.divider()

        # ── Vocabulary Vault ──────────────────────────────────────────────
        st.markdown('<div class="section-header">📚 Vocabulary Vault</div>', unsafe_allow_html=True)

        vocab_svc = _get_vocab_svc()
        word_count = vocab_svc.get_word_count()

        st.markdown(
            f'<div style="margin-bottom:0.8rem;">'
            f'<span class="stat-badge">📖 {word_count} 個單字</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Search box
        search_q = st.text_input(
            "🔎 搜尋單字",
            placeholder="輸入詞彙、原形或翻譯…",
            key="vocab_search_input",
            label_visibility="collapsed",
        )

        # Word list
        if word_count == 0:
            st.markdown(
                '<div style="color:#5a6078; font-size:0.82rem; padding:0.5rem 0;">'
                '尚未收錄任何單字。<br>點擊日文字幕中的詞彙即可加入！'
                '</div>',
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
                pos_esc = html.escape(entry.part_of_speech) if entry.part_of_speech else ""

                ex_html = ""
                if ex_src:
                    ex_src_esc = html.escape(ex_src)
                    ex_tgt_html = f'<div class="vocab-ex-tgt">🇹🇼 {html.escape(ex_tgt)}</div>' if ex_tgt else ""
                    ex_html = (
                        f'<div class="vocab-card-section">'
                        f'<div class="vocab-card-label">所屬例句</div>'
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
                        if st.button("✕", key=f"del_vocab_{entry.id}", help=f"刪除「{entry.surface_form}」"):
                            vocab_svc.delete_word(entry.id)
                            st.session_state.saved_surfaces.discard(entry.surface_form)
                            st.toast(f"🗑️ 已從單字庫刪除：{entry.surface_form}", icon="✅")
                            st.rerun()

                    st.markdown(
                        f'<div class="vocab-card-body">'
                        f'  <div class="vocab-meta-row">'
                        f'    <span class="vocab-card-label">原型：</span>'
                        f'    <span class="vocab-card-val">{base_esc}</span>'
                        f'  </div>'
                        f'  <div class="vocab-meta-row">'
                        f'    <span class="vocab-card-label">讀音（片假名）：</span>'
                        f'    <span class="vocab-card-val reading">{reading_esc}</span>'
                        f'  </div>'
                        f'</div>'
                        f'{ex_html}',
                        unsafe_allow_html=True,
                    )

            if len(results) > 50:
                st.caption(f"顯示前 50 筆，共 {len(results)} 筆結果")

            # ── Clear All Words Button ─────────────────────────────────────────
            st.markdown('<div style="height:0.3rem;"></div>', unsafe_allow_html=True)
            if not st.session_state.get("confirm_clear_vocab", False):
                if st.button("🗑️ 清空所有單字", use_container_width=True, help="清空單字庫內所有單字"):
                    st.session_state.confirm_clear_vocab = True
                    st.rerun()
            else:
                st.warning("⚠️ 確定要清空所有單字嗎？此動作無法復原！")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("⚠️ 確認清空", type="primary", use_container_width=True):
                        vocab_svc.delete_all_words()
                        st.session_state.saved_surfaces.clear()
                        st.session_state.confirm_clear_vocab = False
                        st.toast("🗑️ 已清空所有單字庫資料！", icon="✅")
                        st.rerun()
                with col_no:
                    if st.button("取消", use_container_width=True):
                        st.session_state.confirm_clear_vocab = False
                        st.rerun()

        st.divider()

        # Export button
        if word_count > 0:
            anki_bytes = _build_anki_bytes()
            st.download_button(
                label="⬇️  匯出 Anki (.tsv)",
                data=anki_bytes,
                file_name="bilingual_sync_anki.tsv",
                mime="text/tab-separated-values",
                use_container_width=True,
                help="下載 Anki 匯入格式（Tab 分隔，UTF-8 BOM）",
            )
        else:
            st.button(
                "⬇️  匯出 Anki (.tsv)",
                disabled=True,
                use_container_width=True,
                help="尚無單字可匯出",
            )

        st.markdown(
            '<div style="font-size:0.75rem; color:#8892a0; margin-top:0.45rem; line-height:1.45;">'
            '💡 提示：匯出的 .tsv 檔符合 Anki 欄位標準，開啟 Anki 點選「匯入檔案」即可直接生成單字牌組。'
            '</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main area: Empty State
# ─────────────────────────────────────────────────────────────────────────────

def _render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">🎬</div>
            <h3>歡迎使用 BilingualSync</h3>
            <p>請在左側側邊欄上傳<strong>日文</strong>與<strong>中文</strong>字幕檔，<br>
               點擊「解析並對齊字幕」後即可開始互動學習。</p>
        </div>

        <div style="display:flex; gap:1rem; margin-top:2rem; justify-content:center; flex-wrap:wrap;">
            <div class="glass-card" style="min-width:200px; text-align:center; flex:1;">
                <div style="font-size:1.8rem; margin-bottom:0.4rem;">🔍</div>
                <div style="font-weight:600; color:#A29BFE;">智慧字幕對齊</div>
                <div style="font-size:0.8rem; color:#5a6078; margin-top:0.3rem;">滑動視窗時間戳重疊演算法<br>自動匹配雙語句對</div>
            </div>
            <div class="glass-card" style="min-width:200px; text-align:center; flex:1;">
                <div style="font-size:1.8rem; margin-bottom:0.4rem;">🎯</div>
                <div style="font-weight:600; color:#A29BFE;">一鍵加入單字庫</div>
                <div style="font-size:0.8rem; color:#5a6078; margin-top:0.3rem;">點擊任意日文詞彙<br>自動解析原形、讀音與詞性</div>
            </div>
            <div class="glass-card" style="min-width:200px; text-align:center; flex:1;">
                <div style="font-size:1.8rem; margin-bottom:0.4rem;">📤</div>
                <div style="font-weight:600; color:#A29BFE;">Anki 匯出</div>
                <div style="font-size:0.8rem; color:#5a6078; margin-top:0.3rem;">完整例句上下文<br>一鍵匯出 TSV 單字卡</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main area: Bilingual Reader
# ─────────────────────────────────────────────────────────────────────────────

# Part-of-speech display rules
_POS_STYLE = {
    "動詞":    ("🔵", "#6C63FF"),   # verb  – purple
    "名詞":    ("🟢", "#00CEC9"),   # noun  – teal
    "形容詞":  ("🟡", "#FDCB6E"),   # i-adj – yellow
    "形容動詞":("🟠", "#E17055"),   # na-adj – orange
    "副詞":    ("🔴", "#FF7675"),   # adverb – red
}
# POS tags we don't want as clickable buttons (grammatical particles etc.)
_SKIP_BUTTON_POS = frozenset({"助詞", "助動詞", "記号", "BOS/EOS", "接頭詞", "接尾"})


def _render_reader() -> None:
    pairs = st.session_state.aligned_pairs
    tokenizer = _init_tokenizer()
    vocab_svc = _get_vocab_svc()

    # Header
    n_pairs = len(pairs)
    n_matched = sum(1 for p in pairs if p.target is not None)
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(
            '<div style="font-size:1.4rem; font-weight:700; color:#A29BFE; margin-bottom:0.3rem;">'
            '🎬 雙語字幕對照閱讀器'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_h2:
        st.markdown(
            f'<div style="text-align:right; padding-top:0.3rem;">'
            f'<span class="stat-badge">📝 {n_matched}/{n_pairs} 句</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="font-size:0.78rem; color:#5a6078; margin-bottom:1rem;">'
        '💡 點擊日文詞彙即可加入單字庫・顏色代表詞性：'
        '🔵動詞 🟢名詞 🟡形容詞 🟠形容動詞 🔴副詞'
        '</div>',
        unsafe_allow_html=True,
    )

    # Render each sentence pair
    for pair in pairs:
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
                        f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:0.35rem;">'
                        f'<span class="subtitle-index">#{src.index}</span>'
                        f'<span class="ts-chip">{ts_start}→{ts_end}</span>'
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
                                    emoji, color = _POS_STYLE.get(pos, ("⚪", "#8892a0"))
                                    is_saved = surf in st.session_state.saved_surfaces
                                    btn_label = f"{'✓ ' if is_saved else ''}{surf}"
                                    btn_key = f"{key_prefix}_{btn_idx}"
                                    btn_type = "primary" if is_saved else "secondary"
                                    if st.button(
                                        btn_label,
                                        key=btn_key,
                                        help=f"{emoji} {pos}  |  原形：{base}",
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
                            '<div style="display:flex; align-items:center; gap:8px; margin-bottom:0.35rem; opacity:0.35;">'
                            '<span class="subtitle-index">–</span>'
                            '</div>'
                            '<div class="zh-text" style="opacity:0.35; margin-top:0.35rem;">（無對應中文字幕）</div>',
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
        st.toast(f"✅ 加入單字庫：**{surface}**{' → ' + detail if detail else ''}", icon="🎌")
        st.rerun()
    except Exception as exc:
        st.toast(f"⚠️ 儲存失敗：{exc}", icon="❌")


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

    # Sidebar
    _render_sidebar()

    # Main content
    st.markdown(
        '<div style="height:0.2rem;"></div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.aligned_pairs:
        _render_empty_state()
    else:
        _render_reader()


if __name__ == "__main__":
    main()
