# 🎬 BilingualSync

<div align="center">

**AI-Driven Bilingual Subtitle Transcription, Alignment & Language Learning Assistant**  
基於 Whisper 與語意對齊的 AI 雙語字幕轉錄與日語自主學習輔助工具

[English](#-english) | [繁體中文](#-繁體中文)

</div>

---

## 🌐 English

### Overview
**BilingualSync** is a lightweight desktop application designed for bilingual video transcription and context-based language learning. Powered by `faster-whisper`, it generates Japanese transcriptions and aligns them with Traditional Chinese translations.

### Key Features
- **Local Whisper ASR**: Fast and accurate transcription using `faster-whisper` (medium model, CPU int8 quantization).
- **Resilient Translation Pipeline**: Multi-provider fallback chain (Google Translator → MyMemory) to bypass API 429 rate limits.
- **Morphological Token Chips**: Japanese morphological analysis allows interactive token-by-token alignment without layout distortion.
- **Clean Dual-Column UI**: Fully responsive dual-column layout with customized Flexbox styling for seamless reading.

### Quick Start
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/PeEnLin/BilingualSync.git](https://github.com/PeEnLin/BilingualSync.git)
   cd BilingualSync

   ---

## 🇹🇼 繁體中文

### 專案簡介
**BilingualSync** 是一套專為雙語影音轉錄與語言沉浸式學習打造的桌面端工具。核心採用 `faster-whisper`，能高效完成日語語音轉錄，並自動對齊繁體中文字幕與語法標籤，提供舒適的閱讀與學習介面。

### 核心特性
- **本機高效 ASR 辨識**：採用 `faster-whisper`（medium 模型 + CPU int8 量化），維持高辨識率同時兼顧系統資源。
- **高容錯翻譯管線**：實作多 Provider 自動容錯鏈（遭遇 429 頻率限制自動切換備援），杜絕請求中斷與卡死。
- **單字晶片流式排版**：整合日語形態素分析，以自適應 Flexbox 晶片呈現詞性標籤，告別欄位擠壓變形。
- **雙語即時對照 UI**：深度微調 Streamlit 原生樣式，提供排版呼吸感極佳的左右對照閱讀器。

### 快速開始
1. **複製專案庫**：
   ```bash
   git clone [https://github.com/PeEnLin/BilingualSync.git](https://github.com/PeEnLin/BilingualSync.git)
   cd BilingualSync