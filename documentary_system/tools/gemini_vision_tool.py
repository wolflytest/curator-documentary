"""
Gemini Vision ile görsel içerik analizi aracı.
"""
import json
import logging
import re
from pathlib import Path

from crewai.tools import BaseTool
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config import GEMINI_API_KEY, GEMINI_MODEL_FALLBACK, GEMINI_MODEL_PRIMARY

log = logging.getLogger(__name__)

_gemini_client = genai.Client(api_key=GEMINI_API_KEY)

_MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _is_retryable(exc: BaseException) -> bool:
    """503 ve 429 hatalarında retry yap."""
    if isinstance(exc, (genai_errors.ServerError, genai_errors.ClientError)):
        return getattr(exc, "status_code", 0) in (429, 503)
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=5, max=20),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_gemini_vision(model: str, parts: list) -> str:
    """Gemini'ye görsel + metin isteği gönder."""
    response = _gemini_client.models.generate_content(
        model=model,
        contents=parts,
        config=genai_types.GenerateContentConfig(max_output_tokens=1024),
    )
    return response.text.strip()


class GeminiVisionTool(BaseTool):
    name: str = "Görsel Analiz Et"
    description: str = """
    Bir görseli veya video frame'ini analiz eder.
    Input: JSON string {
        "image_path": "/tmp/...",
        "scene_text": "Osmanlı ordusu İstanbul'u kuşatıyor",
        "keywords": ["ottoman", "siege", "1453"],
        "visual_context": "Önceki sahneler sepia tonlu tarihi gravürler kullandı"
    }
    Output: JSON {
        "alakali": bool,
        "alakalilik_skoru": 1-10,
        "watermark_var": bool,
        "kalite_skoru": 1-10,
        "icerik": str,
        "dominant_renkler": [str],
        "donem_uyumu": bool,
        "kullanilabilir": bool,
        "red_nedeni": str
    }
    """

    def _run(self, input_str: str) -> str:
        """Görseli Gemini ile analiz et, JSON sonuç döndür."""
        try:
            data = json.loads(input_str)
            image_path = Path(data.get("image_path", ""))
            scene_text = data.get("scene_text", "")
            keywords = data.get("keywords", [])
            visual_context = data.get("visual_context", "")
        except (json.JSONDecodeError, KeyError) as exc:
            log.error("GeminiVisionTool geçersiz input: %s", exc)
            return json.dumps({"kullanilabilir": False, "red_nedeni": f"Geçersiz input: {exc}"})

        if not image_path.exists():
            log.error("Görsel dosyası bulunamadı: %s", image_path)
            return json.dumps({"kullanilabilir": False, "red_nedeni": "Dosya bulunamadı"})

        mime_type = _MIME_MAP.get(image_path.suffix.lower(), "image/jpeg")

        prompt = f"""Bu görseli analiz et ve aşağıdaki JSON formatında yanıt ver:

Sahne metni: {scene_text}
Aranan anahtar kelimeler: {', '.join(keywords)}
Görsel bağlam (önceki sahneler): {visual_context or 'yok'}

Değerlendirme kriterleri:
1. Görsel, sahne metnini ne kadar destekliyor? (1-10)
2. Watermark veya logo var mı?
3. Görsel kalitesi nasıl? (1-10)
4. Dominant renkler neler?
5. Dönem/tema uyumu var mı?

Yanıtı SADECE şu JSON formatında ver (başka hiçbir şey yazma):
{{
  "alakali": true,
  "alakalilik_skoru": 7,
  "watermark_var": false,
  "kalite_skoru": 8,
  "icerik": "görselde ne var kısa açıklama",
  "dominant_renkler": ["renk1", "renk2"],
  "donem_uyumu": true,
  "kullanilabilir": true,
  "red_nedeni": ""
}}"""

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            parts = [
                prompt,
                genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ]

            try:
                raw = _call_gemini_vision(GEMINI_MODEL_PRIMARY, parts)
            except genai_errors.ClientError as exc:
                if getattr(exc, "status_code", 0) in (404, 429):
                    raw = _call_gemini_vision(GEMINI_MODEL_FALLBACK, parts)
                else:
                    raise

            # JSON çıkart
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return match.group(0)
            return raw

        except Exception as exc:
            log.error("GeminiVisionTool analiz hatası: %s", exc)
            return json.dumps({
                "alakali": False,
                "alakalilik_skoru": 0,
                "watermark_var": False,
                "kalite_skoru": 0,
                "icerik": "",
                "dominant_renkler": [],
                "donem_uyumu": False,
                "kullanilabilir": False,
                "red_nedeni": str(exc),
            })


if __name__ == "__main__":
    print("✅ GeminiVisionTool modülü yüklendi")
    tool = GeminiVisionTool()
    print(f"   Tool adı: {tool.name}")
