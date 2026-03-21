"""
Medya indirme aracı: URL'den dosyayı /tmp/curator_docs/ altına indirir.
WikimediaTool/PexelsTool çıktısındaki download_url'i local path'e çevirir.
"""
import hashlib
import json
import logging
import mimetypes
from pathlib import Path

import httpx
from crewai.tools import BaseTool

log = logging.getLogger(__name__)

_DOWNLOAD_DIR = Path("/tmp/curator_docs/media")
_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

_HEADERS = {
    "User-Agent": "curator-documentary-bot/1.0 (https://github.com/wolflytest/curator-documentary)"
}


class MediaDownloadTool(BaseTool):
    name: str = "Medya İndir"
    description: str = """
    Bir URL'deki fotoğraf veya videoyu /tmp/curator_docs/media/ altına indirir.
    Input: JSON string {"url": "https://...", "media_type": "photo|video"}
    Output: JSON {"local_path": "/tmp/...", "success": bool, "error": str}
    WikimediaTool veya PexelsTool çıktısındaki download_url'i GeminiVisionTool için local path'e çevirir.
    """

    def _run(self, input_str: str) -> str:
        """URL'yi indir, local path döndür."""
        try:
            data = json.loads(input_str)
            url = data.get("url", "")
            media_type = data.get("media_type", "photo")
        except (json.JSONDecodeError, KeyError) as exc:
            log.error("MediaDownloadTool geçersiz input: %s", exc)
            return json.dumps({"local_path": "", "success": False, "error": str(exc)})

        if not url:
            return json.dumps({"local_path": "", "success": False, "error": "URL boş"})

        # URL'den tekil dosya adı üret (aynı URL iki kez indirilmesin)
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]

        # Uzantıyı URL'den tahmin et
        ext = _guess_extension(url, media_type)
        local_path = _DOWNLOAD_DIR / f"{url_hash}{ext}"

        # Zaten indirilmişse direkt döndür
        if local_path.exists() and local_path.stat().st_size > 1000:
            log.info("Önbellekten kullanılıyor: %s", local_path.name)
            return json.dumps({"local_path": str(local_path), "success": True, "error": ""})

        try:
            with httpx.Client(timeout=30, headers=_HEADERS, follow_redirects=True) as client:
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()

                    # Content-Type'dan uzantı güncelle
                    content_type = resp.headers.get("content-type", "")
                    real_ext = _ext_from_content_type(content_type) or ext
                    if real_ext != ext:
                        local_path = _DOWNLOAD_DIR / f"{url_hash}{real_ext}"

                    with open(local_path, "wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=8192):
                            f.write(chunk)

            size_kb = local_path.stat().st_size // 1024
            log.info("İndirildi: %s (%d KB)", local_path.name, size_kb)
            return json.dumps({"local_path": str(local_path), "success": True, "error": ""})

        except Exception as exc:
            log.error("MediaDownloadTool indirme hatası (%s): %s", url[:80], exc)
            if local_path.exists():
                local_path.unlink(missing_ok=True)
            return json.dumps({"local_path": "", "success": False, "error": str(exc)})


def _guess_extension(url: str, media_type: str) -> str:
    """URL'nin son segmentinden uzantı tahmin et."""
    path_part = url.split("?")[0].rstrip("/")
    suffix = Path(path_part).suffix.lower()
    if suffix in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        return suffix
    if suffix in (".mp4", ".webm", ".mov"):
        return suffix
    return ".jpg" if media_type == "photo" else ".mp4"


def _ext_from_content_type(content_type: str) -> str:
    """Content-Type header'ından dosya uzantısı üret."""
    ct = content_type.split(";")[0].strip()
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "video/mp4": ".mp4",
        "video/webm": ".webm",
    }
    return mapping.get(ct, "")


if __name__ == "__main__":
    import json
    tool = MediaDownloadTool()
    # Pexels örnek fotoğraf
    test_url = "https://images.pexels.com/photos/71241/pexels-photo-71241.jpeg?w=400"
    r = tool._run(json.dumps({"url": test_url, "media_type": "photo"}))
    result = json.loads(r)
    print(f"✅ MediaDownloadTool: success={result['success']}, path={result['local_path']}")
