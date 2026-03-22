"""
Ortam değişkenlerini .env dosyasından yükler.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasını proje kökünden yükle
load_dotenv(Path(__file__).parent / ".env")

TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_USER_ID: int = int(os.environ["TELEGRAM_USER_ID"])
GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
PEXELS_API_KEY: str = os.environ["PEXELS_API_KEY"]
PIXABAY_API_KEY: str = os.getenv("PIXABAY_API_KEY", "")
SILICONFLOW_API_KEY: str = os.getenv("SILICONFLOW_API_KEY", "")
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")

# Geçici dosyalar için dizin (700: sadece bot process erişebilir)
TMP_DIR = Path("/tmp/curator")
TMP_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)

# Veritabanı dosyası
DB_PATH = Path(__file__).parent / "curator.db"

# Gemini'ye gönderilecek maksimum frame sayısı
MAX_FRAMES = 16

# Günlük özet saati (24 saat formatı)
SUMMARY_HOUR = 21
SUMMARY_MINUTE = 0

# Instagram cookie dosyası (sunucuda mevcutsa kullanılır)
COOKIE_FILE = Path("/home/ubuntu/curator/instagram_cookies.txt")

# Gemini model sıralaması: önce primary denenir, 404/429 alınırsa fallback
GEMINI_MODEL_PRIMARY  = "gemini-3.1-flash-lite-preview"
GEMINI_MODEL_FALLBACK = "gemini-2.5-flash-lite"

# Sohbet için DB'den kaç kayıt çekileceği
CHAT_CONTEXT_LIMIT = 50

# OpenClaw bilgi tabanı dizini
OPENCLAW_DIR = Path(__file__).parent / "openclaw_knowledge"
OPENCLAW_DIR.mkdir(exist_ok=True)

# Kokoro ONNX sesleri (yerel, hızlı) — kokoro-v1.0.int8.onnx gerekli
KOKORO_VOICES: dict[str, dict[str, str]] = {
    "en": {
        "bm_george":  "🎙 George (İngiliz Erkek) - Belgesel",
        "bm_daniel":  "🎙 Daniel (İngiliz Erkek) - Haber",
        "bm_lewis":   "🎙 Lewis (İngiliz Erkek) - Dramatik",
        "am_michael": "🎙 Michael (ABD Erkek) - Nötr",
        "am_liam":    "🎙 Liam (ABD Erkek) - Genç",
        "bf_emma":    "🎙 Emma (İngiliz Kadın) - Zarif",
        "af_bella":   "🎙 Bella (ABD Kadın) - Sıcak",
        "af_nova":    "🎙 Nova (ABD Kadın) - Enerjik",
    },
}

# Microsoft edge_tts sesleri (ücretsiz cloud, İngilizce)
EDGE_TTS_VOICES: dict[str, dict[str, str]] = {
    "en": {
        "en-GB-RyanNeural":      "☁️ Ryan (İngiliz Erkek) - Belgesel",
        "en-GB-ThomasNeural":    "☁️ Thomas (İngiliz Erkek) - Haber",
        "en-US-GuyNeural":       "☁️ Guy (ABD Erkek) - Nötr",
        "en-US-ChristopherNeural": "☁️ Christopher (ABD Erkek) - Otoriter",
        "en-GB-SoniaNeural":     "☁️ Sonia (İngiliz Kadın) - Zarif",
        "en-US-JennyNeural":     "☁️ Jenny (ABD Kadın) - Sıcak",
        "en-US-AriaNeural":      "☁️ Aria (ABD Kadın) - Enerjik",
    },
}

# ElevenLabs TTS sesleri (yüksek kalite cloud, ELEVENLABS_API_KEY gerekli)
ELEVENLABS_VOICES: dict[str, dict[str, str]] = {
    "en": {
        "elevenlabs:Adam":    "⭐ Adam (Erkek) - Belgesel Kalite",
        "elevenlabs:Antoni":  "⭐ Antoni (Erkek) - Dramatik",
        "elevenlabs:Arnold":  "⭐ Arnold (Erkek) - Güçlü",
        "elevenlabs:Rachel":  "⭐ Rachel (Kadın) - Doğal",
        "elevenlabs:Domi":    "⭐ Domi (Kadın) - Enerjik",
    },
}

# Dashboard için birleşik TTS_VOICES (hepsi bir arada)
TTS_VOICES: dict[str, dict[str, str]] = {
    "en": {
        **KOKORO_VOICES["en"],
        **EDGE_TTS_VOICES["en"],
        **ELEVENLABS_VOICES["en"],
        "chatterbox:default:Default Voice-Neutral": "🔬 Chatterbox (Varsayılan)",
        "chatterbox:clone:Voice Clone-Custom":      "🔬 Chatterbox (Ses Klonlama)",
        "siliconflow:FunAudioLLM/CosyVoice2-0.5B:alex-Male":   "☁️ SiliconFlow Alex (Erkek)",
        "siliconflow:FunAudioLLM/CosyVoice2-0.5B:anna-Female":  "☁️ SiliconFlow Anna (Kadın)",
    },
    "tr": {
        "gtts_tr": "🎙 Türkçe (gTTS)",
    },
}

# Aspect ratio seçenekleri (MPT-Extended VideoAspect)
VIDEO_ASPECTS: dict[str, str] = {
    "16:9": "🖥️ Yatay (16:9) — 1920×1080",
    "9:16": "📱 Dikey (9:16) — 1080×1920",
    "1:1":  "⬛ Kare (1:1) — 1080×1080",
}

# Belgesel hedef süreleri: str_saniye → etiket
VIDEO_DURATIONS: dict[str, str] = {
    "30":  "30 saniye (Test)",
    "120": "2 dakika (Kısa)",
    "300": "5 dakika (Orta)",
    "600": "10 dakika (Standart)",
    "900": "15 dakika (Uzun)",
}
