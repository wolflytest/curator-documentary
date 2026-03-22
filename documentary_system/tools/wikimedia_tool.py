"""
Wikimedia Commons'ta tarihi fotoğraf ve görsel arama aracı.
Gevşetilmiş filtreler ile daha fazla sonuç döndürür.
"""
import json
import logging
import time

import httpx
from crewai.tools import BaseTool

log = logging.getLogger(__name__)

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
_HEADERS = {
    "User-Agent": "curator-documentary-bot/1.0 (https://github.com/wolflytest/curator-documentary)"
}


class WikimediaTool(BaseTool):
    name: str = "Wikimedia Görsel Ara"
    description: str = """
    Wikimedia Commons'ta tarihi fotoğraf ve görsel arar.
    Input: JSON string {"keyword": "Ottoman Empire 1900", "limit": 5}
    Output: JSON list of {title, url, download_url, license, width, height}
    """

    def _run(self, input_str: str) -> str:
        """Wikimedia Commons'ta görsel ara, birden fazla strateji dene."""
        try:
            try:
                data = json.loads(input_str)
            except (json.JSONDecodeError, ValueError):
                data = {"keyword": input_str.strip(), "limit": 5}

            keyword = data.get("keyword", "")
            limit = min(int(data.get("limit", 5)), 20)
        except Exception as exc:
            log.error("WikimediaTool geçersiz input: %s", exc)
            return json.dumps([])

        if not keyword:
            return json.dumps([])

        # Birden fazla arama stratejisi dene
        search_queries = [
            keyword,
            f"{keyword} historical",
            f"{keyword} ancient",
        ]

        all_results = []

        for query in search_queries:
            if len(all_results) >= limit:
                break

            params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": "6",
                "gsrlimit": str(limit * 2),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "format": "json",
            }

            try:
                with httpx.Client(timeout=15, headers=_HEADERS) as client:
                    resp = client.get(WIKIMEDIA_API, params=params)
                    resp.raise_for_status()
                    result = resp.json()
            except Exception as exc:
                log.error("Wikimedia API hatası (%s): %s", query, exc)
                continue

            pages = result.get("query", {}).get("pages", {})

            for page in pages.values():
                if len(all_results) >= limit:
                    break

                infos = page.get("imageinfo", [])
                if not infos:
                    continue
                info = infos[0]

                # Sadece JPEG/PNG
                mime = info.get("mime", "")
                if mime not in ("image/jpeg", "image/png"):
                    continue

                # Minimum çözünürlük (gevşetildi: 400px)
                width = info.get("width", 0)
                if width < 400:
                    continue

                # Lisans kontrolü (gevşetildi: CC, public domain, GFDL)
                ext_meta = info.get("extmetadata", {})
                license_name = ext_meta.get("LicenseShortName", {}).get("value", "")
                if license_name and not any(
                    x in license_name.upper()
                    for x in ["CC", "PUBLIC", "PD", "FREE", "GFDL"]
                ):
                    continue

                # Aynı URL'yi tekrar ekleme
                download_url = info.get("url", "")
                if not download_url or any(r["download_url"] == download_url for r in all_results):
                    continue

                all_results.append({
                    "title": page.get("title", ""),
                    "url": info.get("descriptionurl", ""),
                    "download_url": download_url,
                    "license": license_name or "unknown",
                    "width": width,
                    "height": info.get("height", 0),
                })

                time.sleep(0.3)

        log.info("Wikimedia araması: %d sonuç (%s)", len(all_results), keyword)
        return json.dumps(all_results, ensure_ascii=False)


if __name__ == "__main__":
    tool = WikimediaTool()
    r = tool._run(json.dumps({"keyword": "Roman Empire", "limit": 3}))
    items = json.loads(r)
    print(f"✅ Wikimedia: {len(items)} sonuç")
    for item in items[:2]:
        print(f"   - {item['title'][:50]} {item['width']}px [{item['license']}]")
