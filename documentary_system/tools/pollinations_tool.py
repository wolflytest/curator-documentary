"""
Pollinations.AI text-to-image aracı — ücretsiz, API key gerektirmez.

Kullanım: https://image.pollinations.ai/prompt/{prompt}
Parametreler: width, height, model, seed, nologo=true
"""
import hashlib
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path

import config

log = logging.getLogger(__name__)

_CACHE_DIR = config.TMP_DIR / "pollinations_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)

# Pollinations modelleri: flux (en iyi kalite), turbo (hız)
_DEFAULT_MODEL = "flux"
_BASE_URL = "https://image.pollinations.ai/prompt"


def generate_image(
    prompt: str,
    width: int = 1920,
    height: int = 1080,
    model: str = _DEFAULT_MODEL,
    seed: int | None = None,
    max_retries: int = 3,
) -> dict:
    """
    Pollinations.AI ile metin→görüntü üret.

    Returns:
        {"success": True, "local_path": str, "prompt": str, "source": "pollinations"}
        veya {"success": False, "error": str}
    """
    # Cache key — aynı prompt/boyut için yeniden üretme
    cache_key = hashlib.md5(f"{prompt}|{width}|{height}|{model}|{seed}".encode()).hexdigest()[:16]
    cached_path = _CACHE_DIR / f"{cache_key}.jpg"

    if cached_path.exists() and cached_path.stat().st_size > 10_000:
        log.info("Pollinations cache hit: %s", cached_path.name)
        return {"success": True, "local_path": str(cached_path), "prompt": prompt, "source": "pollinations"}

    # URL oluştur
    encoded_prompt = urllib.parse.quote(prompt)
    _NEGATIVE = (
        "cartoon,illustration,anime,painting,drawing,text,watermark,logo,"
        "low quality,blurry,distorted,oversaturated,fake,cgi,3d render"
    )
    params = [
        f"width={width}",
        f"height={height}",
        f"model={model}",
        "nologo=true",
        "enhance=false",
        f"negative_prompt={urllib.parse.quote(_NEGATIVE)}",
    ]
    if seed is not None:
        params.append(f"seed={seed}")

    url = f"{_BASE_URL}/{encoded_prompt}?{'&'.join(params)}"

    for attempt in range(max_retries):
        try:
            log.info("Pollinations görüntü üretiliyor (deneme %d/%d): %s...", attempt + 1, max_retries, prompt[:60])

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "curator-documentary/1.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                content_type = resp.headers.get("Content-Type", "")
                if "image" not in content_type:
                    log.warning("Pollinations beklenmeyen content-type: %s", content_type)
                    continue

                data = resp.read()

            if len(data) < 5_000:
                log.warning("Pollinations çok küçük yanıt (%d bytes), yeniden deneniyor...", len(data))
                time.sleep(2)
                continue

            cached_path.write_bytes(data)
            log.info("Pollinations görüntü kaydedildi: %s (%d KB)", cached_path.name, len(data) // 1024)
            return {
                "success": True,
                "local_path": str(cached_path),
                "prompt": prompt,
                "source": "pollinations",
                "model": model,
            }

        except Exception as exc:
            log.warning("Pollinations hata (deneme %d): %s", attempt + 1, exc)
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))

    return {"success": False, "error": "Tüm denemeler başarısız", "prompt": prompt}


def aspect_to_resolution(aspect: str) -> tuple[int, int]:
    """Video aspect ratio'ya göre Pollinations için en-yükseklik döndür."""
    mapping = {
        "16:9": (1920, 1080),
        "9:16": (1080, 1920),
        "1:1":  (1080, 1080),
    }
    return mapping.get(aspect, (1920, 1080))
