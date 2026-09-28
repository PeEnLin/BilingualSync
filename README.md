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