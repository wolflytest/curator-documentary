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

# TTS ses seçenekleri: dil → {ses_kodu: etiket}
TTS_VOICES: dict[str, dict[str, str]] = {
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
    "tr": {
        "gtts_tr": "🎙 Türkçe (gTTS)",
    },
}

# Belgesel hedef süreleri: str_saniye → etiket
VIDEO_DURATIONS: dict[str, str] = {
    "30":  "30 saniye (Test)",
    "120": "2 dakika (Kısa)",
    "300": "5 dakika (Orta)",
    "600": "10 dakika (Standart)",
    "900": "15 dakika (Uzun)",
}
