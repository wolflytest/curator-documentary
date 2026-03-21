"""
Wikimedia Commons'ta tarihi fotoğraf ve görsel arama aracı.
"""
import json
import logging
import time

import httpx
from crewai.tools import BaseTool

log = logging.getLogger(__name__)

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"


class WikimediaTool(BaseTool):
    name: str = "Wikimedia Görsel Ara"
    description: str = """
    Wikimedia Commons'ta tarihi fotoğraf ve görsel arar.
    Input: JSON string {"keyword": "Ottoman Empire 1900", "limit": 5}
    Output: JSON list of {title, url, download_url, license, width, height}
    Sadece CC lisanslı, yüksek çözünürlüklü görseller döndürür.
    """

    def _run(self, input_str: str) -> str:
        """Wikimedia Commons'ta görsel ara, CC lisanslı yüksek çözünürlüklü sonuçları döndür."""
        try:
            data = json.loads(input_str)
            keyword = data.get("keyword", "")
            limit = min(int(data.get("limit", 5)), 20)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            log.error("WikimediaTool geçersiz input: %s", exc)
            return json.dumps([])

        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"{keyword} historical photograph",
            "gsrnamespace": "6",
            "gsrlimit": str(limit * 3),  # Filtreleme için fazla al
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "format": "json",
        }

        headers = {
            "User-Agent": "curator-documentary-bot/1.0 (https://github.com/wolflytest/curator-documentary)"
        }
        try:
            with httpx.Client(timeout=15, headers=headers) as client:
                resp = client.get(WIKIMEDIA_API, params=params)
                resp.raise_for_status()
                result = resp.json()
        except Exception as exc:
            log.error("Wikimedia API hatası: %s", exc)
            return json.dumps([])

        pages = result.get("query", {}).get("pages", {})
        results = []

        for page in pages.values():
            infos = page.get("imageinfo", [])
            if not infos:
                continue
            info = infos[0]

            # Sadece JPEG/PNG
            mime = info.get("mime", "")
            if mime not in ("image/jpeg", "image/png"):
                continue

            # Minimum çözünürlük
            width = info.get("width", 0)
            if width < 800:
                continue

            # CC lisans kontrolü
            ext_meta = info.get("extmetadata", {})
            license_name = ext_meta.get("LicenseShortName", {}).get("value", "")
            if not license_name.startswith("CC"):
                continue

            results.append({
                "title": page.get("title", ""),
                "url": info.get("descriptionurl", ""),
                "download_url": info.get("url", ""),
                "license": license_name,
                "width": width,
                "height": info.get("height", 0),
            })

            if len(results) >= limit:
                break

            time.sleep(0.5)  # Rate limit önlemi

        log.info("Wikimedia araması tamamlandı: %d sonuç (%s)", len(results), keyword)
        return json.dumps(results, ensure_ascii=False)


if __name__ == "__main__":
    tool = WikimediaTool()
    r = tool._run(json.dumps({"keyword": "Ottoman Empire Constantinople 1453", "limit": 2}))
    items = json.loads(r)
    print(f"✅ Wikimedia: {len(items)} sonuç")
    for item in items:
        print(f"   - {item['title']} ({item['width']}x{item['height']}) [{item['license']}]")
