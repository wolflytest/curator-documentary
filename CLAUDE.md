# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proje

`curator` — iki bağımsız sistemden oluşur:

1. **Curator Bot**: Instagram/TikTok/YouTube/Twitter linklerini Telegram üzerinden alır, analiz eder, gece 21:00'de özet gönderir.
2. **Documentary System**: CrewAI ajan ekipleriyle otomatik belgesel video üretir. Telegram botu ve Streamlit dashboard üzerinden tetiklenir.

## Komutlar

```bash
# Sanal ortamı etkinleştir (her oturumda)
source venv/bin/activate

# Botu başlat
python main.py

# Streamlit dashboard'u başlat (geliştirme - localhost)
streamlit run dashboard.py --server.port 8501

# Oracle Cloud sunucusunda dashboard başlat (0.0.0.0)
bash start_dashboard.sh

# Bağımlılıkları kur
venv/bin/pip install -r requirements.txt

# Import testleri
venv/bin/python -c "import pipeline"
venv/bin/python -c "from documentary_system.orchestrator import run_documentary; print('OK')"

# Belgesel sistemini elle test et
venv/bin/python -c "
import db; db.init_db()
from documentary_system.orchestrator import run_documentary
result = run_documentary('Osmanlı İmparatorluğu', target_duration=30)
print(result)
"

# LLM bağlantısını kontrol et (Ollama/Gemini)
venv/bin/python documentary_system/llm_config.py

# Tek tool'u test et
venv/bin/python documentary_system/tools/pexels_tool.py
venv/bin/python documentary_system/tools/wikimedia_tool.py
```

## Ortam

- Python 3.12, venv: `venv/`
- `.env` dosyası proje kökünde: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_USER_ID`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `PEXELS_API_KEY`
- Geçici dosyalar: `/tmp/curator/<uid>/` ve `/tmp/curator_docs/` — işlem bitince otomatik silinir
- Veritabanı: `curator.db` (SQLite, `db.init_db()` ile otomatik oluşur/migrate edilir)
- Kokoro TTS model dosyaları: proje kökünde `kokoro-v1.0.int8.onnx` ve `voices-v1.0.bin` olmalı
- YouTube cookies: proje kökünde `youtube_cookies.txt` (bot tespitini aşmak için, .gitignore'da)

## Mimari

### Curator Bot

```
main.py       → db.init_db() + bot.build_application() + run_polling()
bot.py        → Telegram handler + APScheduler (21:00 özet) + komutlar
pipeline.py   → URL → yt-dlp → ffmpeg → PySceneDetect → Groq Whisper → Gemini
chat.py       → DB bağlamı + kullanıcı sorusu → Gemini → Türkçe yanıt
db.py         → SQLite: tüm tablo tanımları ve CRUD fonksiyonları
config.py     → .env yükle, tüm sabitler (TTS_VOICES, VIDEO_DURATIONS dahil)
```

#### pipeline.py İşlem Hattı (`pipeline.run(url, note="")`)

1. `detect_platform()` — URL'den Instagram/TikTok/YouTube/Twitter tespit et
2. `download_video()` — yt-dlp ile mp4 indir
3. `extract_audio()` — ffmpeg ile 16kHz mono mp3 ayır
4. `select_frames()` — PySceneDetect (ContentDetector) ile sahne sınırlarından max 8 frame; sahne bulunamazsa eşit aralıklı
5. `transcribe_audio()` — Groq `whisper-large-v3-turbo`
6. `analyse_with_gemini()` — Gemini'ye frame'ler (inline bytes) + transkript + not → JSON → `(analiz_str, öncelik_int)`

Twitter/X'te video yoksa `fetch_tweet_text()` → 4 fallback yöntemi (vxtwitter → fxtwitter → nitter → HTML scrape) → doğrudan Gemini'ye metin.

Gemini analiz JSON alanları: `ozet`, `adimlar`, `araclar`, `rakamlar`, `ipuclari_ve_uyarilar`, `sonuc`, `hedef_kitle`, `eylem_gerektirir_mi`, `oncelik_skoru` (1-10), `oncelik_nedeni`. `max_output_tokens=4096`.

#### Telegram Komutları

| Komut | İşlev |
|-------|-------|
| Link gönder | Pipeline başlatır |
| Link + `#tag` | Tag'li kaydeder |
| Link + metin | Not olarak Gemini'ye gönderir |
| Normal mesaj | `chat.answer()` ile DB'ye bakarak yanıtlar |
| `/ozet` | Günlük özeti hemen gönderir |
| `/soru <metin>` | DB bağlamıyla Gemini cevabı |
| `/istatistik` | Kayıt/platform/tag istatistikleri |
| `/tag <etiket>` | Etikete göre kayıtları listeler |
| `/sil <id>` | Kaydı siler |
| `/belgesel <konu>` | Documentary System'i tetikler |
| `/belgeler` | Belgesel listesi |
| `/ayarlar` | Inline buton menüsü: dil / ses / süre |

`/ayarlar` multi-level inline menü: ana menü → alt menü (dil/ses/süre seçimi) → ◀️ Geri → 💾 Kaydet ve Kapat. Ayarlar `user_settings` tablosunda saklanır, `/belgesel` bunları okur.

---

### Documentary System

```
documentary_system/
├── orchestrator.py           ← Tek giriş noktası: run_documentary()
├── llm_config.py             ← LLM seçimi: Ollama önce, yoksa Gemini fallback
├── state/documentary_state.py ← DocumentaryState + SceneState (JSON serialize)
├── crews/
│   ├── script_crew.py        ← Creator → Optimizer → Critic (3 ajan, sequential)
│   ├── media_crew.py         ← Searcher → Analyzer → ConsistencyAgent (sahne başına)
│   └── qa_crew.py            ← Viewer + TechQA (final kalite kontrolü)
└── tools/
    ├── wikimedia_tool.py     ← Wikimedia Commons API (CC/PD/GFDL lisanslı)
    ├── pexels_tool.py        ← Pexels stok fotoğraf/video API
    ├── ytcc_tool.py          ← YouTube Creative Commons video arama (yt-dlp)
    ├── media_download_tool.py ← download_url → local_path (MD5 önbellek)
    ├── gemini_vision_tool.py  ← local_path → Gemini görsel analiz
    ├── ffmpeg_tool.py        ← ken_burns | clip | subtitle
    └── kokoro_tts_tool.py    ← Kokoro ONNX TTS, fallback: gTTS
```

#### `run_documentary(topic, target_duration, language, voice)` Pipeline

```
1. script_crew.kickoff()      → senaryo JSON (title, scenes[])
2. media_crew × her sahne     → approved_media (local_path)
   ⚠️  70s sleep arası sahne  (Gemini 15 RPM limiti)
3. KokoroTTSTool × her sahne  → tts_path
   language=="en" → kokoro ses kodu, language=="tr" → gtts_tr (gTTS)
4. FFmpegTool × her sahne     → final_clip_path
   photo → ken_burns; video → clip kes; medya yoksa → siyah (lavfi)
5. _concat_scenes()           → tek MP4 (concat demuxer)
6. qa_crew.kickoff()          → qa_score, approved
```

#### DocumentaryState

`pending → scripting → searching → assembling → qa → done` (hata: `error`)

Her adımda `db.update_documentary_status()` ile SQLite'a serialize edilir. `DocumentaryState.to_json()` / `from_json()` ile tam serialize edilebilir.

`get_visual_context_summary()` — son 3 onaylı sahnenin renk/kaynak/tip bilgisini `ConsistencyAgent`'a bağlam olarak verir.

#### CrewAI Kullanım Kuralları

- Her `Task` tek bir `agent` alır (liste değil)
- Önceki görevlere erişim: `context=[task1, task2]`
- `Process.sequential` zorunlu — görevler sırayla çalışır
- LLM JSON çıktısı: `_extract_json()` ile ayrıştırılır (düz JSON, markdown code block ve `{...}` bloğu — üç format toleranslı)

#### Media Pipeline Akışı

Agent veri zinciri: `download_url` → `MediaDownloadTool` → `local_path` → `GeminiVisionTool`

`GeminiVisionTool` yalnızca local dosya yolu kabul eder. `MediaDownloadTool` MD5 hash ile önbellek tutar — aynı URL iki kez indirilmez.

Her tool'un `_run()` başında düz metin / JSON olmayan input için fallback var: `json.loads` başarısız olursa girdi keyword olarak yorumlanır.

#### FFmpeg Detayları

- **Ken Burns**: `scale=8000:-1` → `zoompan` (titreme önlemi), zoom delta `0.0005`, timeout=120s
- **Clip**: `-ss start -t duration -vf scale=1920:1080:force_original_aspect_ratio=decrease,pad`
- **Concat**: `-f concat -safe 0 -c:v copy -c:a aac -b:a 128k -movflags +faststart`

---

### Veritabanı

```
contents          → Telegram'dan kaydedilen içerikler
documentaries     → Belgesel kayıtları (status, script_json, output_path)
documentary_media → Her sahne için seçilen medya
user_settings     → Kullanıcı başına: language, voice, duration
```

`init_db()` idempotent — her başlatmada güvenle çağrılabilir. Eski tablolara `ALTER TABLE` ile sütun ekler.

---

### LLM Yapılandırması

`llm_config.get_llm()`: `localhost:11434`'te Ollama → `qwen2.5:14b` varsa kullan, yoksa `gemini/gemini-3.1-flash-lite-preview` (LiteLLM üzerinden CrewAI için).

Pipeline (`pipeline.py`) için doğrudan `google-genai` SDK: `from google import genai`. CrewAI için `crewai.LLM` + LiteLLM formatı (`gemini/...`).

### Önemli Kararlar

- **Gemini SDK**: `google-generativeai` deprecated; `from google import genai` (google-genai paketi) kullanılıyor
- **Retry**: `analyse_with_gemini()` ve `chat.answer()` üzerinde tenacity ile exponential backoff (5s→10s→20s)
- **Blocking pipeline**: `pipeline.run()` CPU-bound; `run_in_executor(None, ...)` ile çalıştırılıyor
- **Dashboard subprocess**: `dashboard.py` orchestrator'ı `subprocess.Popen` ile çalıştırır — Streamlit event loop çakışmasını önler
- **Türkçe TTS**: Kokoro model dosyaları yoksa gTTS fallback otomatik devreye girer; Türkçe için her zaman `voice="gtts_tr"` kodu kullanılır
- **Rate limit**: Her sahne media crew çağrısı arasında 70s uyku (Gemini 15 RPM)
- **TTS_VOICES yapısı**: `config.TTS_VOICES` iç içe dict: `{dil: {ses_kodu: etiket}}` — 8 İngilizce Kokoro sesi + 1 Türkçe gTTS
- **VIDEO_DURATIONS yapısı**: `config.VIDEO_DURATIONS` string key: `{"30": "30 saniye (Test)", ..., "900": "15 dakika (Uzun)"}`

## MCP Sunucuları

| Sunucu | Komut |
|--------|-------|
| `filesystem` | `npx -y @modelcontextprotocol/server-filesystem /home/emre/curator` |
| `sqlite` | `/home/emre/curator/venv/bin/mcp-server-sqlite --db-path curator.db` |
| `sequentialthinking` | `npx -y @modelcontextprotocol/server-sequential-thinking` |
| `context7` | `npx -y @upstash/context7-mcp` |
