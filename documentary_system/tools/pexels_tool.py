"""
Pexels stok fotoğraf ve video arama aracı.
"""
import json
import logging

import httpx
from crewai.tools import BaseTool

from config import PEXELS_API_KEY

log = logging.getLogger(__name__)

PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"
PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"


class PexelsTool(BaseTool):
    name: str = "Pexels Medya Ara"
    description: str = """
    Pexels'te ücretsiz stok fotoğraf ve video arar.
    Input: JSON string {"keyword": "ancient city ruins", "type": "photo|video", "limit": 5}
    Output: JSON list of {title, url, download_url, type, width, height, duration}
    """

    def _run(self, input_str: str) -> str:
        """Pexels API ile fotoğraf veya video ara."""
        try:
            try:
                data = json.loads(input_str)
            except (json.JSONDecodeError, ValueError):
                data = {"keyword": input_str.strip(), "type": "photo", "limit": 5}

            keyword = data.get("keyword", "")
            media_type = data.get("type", "photo")
            limit = min(int(data.get("limit", 5)), 20)

            if not keyword:
                return json.dumps([])
        except Exception as exc:
            log.error("PexelsTool geçersiz input: %s", exc)
            return json.dumps([])

        headers = {"Authorization": PEXELS_API_KEY}
        params = {"query": keyword, "per_page": limit}

        try:
            api_url = PEXELS_VIDEO_URL if media_type == "video" else PEXELS_PHOTO_URL
            with httpx.Client(timeout=15) as client:
                resp = client.get(api_url, headers=headers, params=params)
                resp.raise_for_status()
                result = resp.json()
        except Exception as exc:
            log.error("Pexels API hatası: %s", exc)
            return json.dumps([])

        results = []
        if media_type == "video":
            for item in result.get("videos", []):
                video_files = item.get("video_files", [])
                hd_files = [f for f in video_files if f.get("quality") == "hd"]
                chosen = hd_files[0] if hd_files else (video_files[0] if video_files else None)
                if not chosen:
                    continue
                results.append({
                    "title": item.get("url", ""),
                    "url": item.get("url", ""),
                    "download_url": chosen.get("link", ""),
                    "type": "video",
                    "width": chosen.get("width", 0),
                    "height": chosen.get("height", 0),
                    "duration": item.get("duration", 0),
                })
        else:
            for item in result.get("photos", []):
                results.append({
                    "title": item.get("alt", f"photo_{item.get('id')}"),
                    "url": item.get("url", ""),
                    "download_url": item.get("src", {}).get("large2x", ""),
                    "type": "photo",
                    "width": item.get("width", 0),
                    "height": item.get("height", 0),
                    "duration": 0,
                })

        log.info("Pexels araması tamamlandı: %d %s (%s)", len(results), media_type, keyword)
        return json.dumps(results, ensure_ascii=False)


if __name__ == "__main__":
    tool = PexelsTool()
    r = tool._run(json.dumps({"keyword": "ancient ruins", "type": "photo", "limit": 2}))
    items = json.loads(r)
    print(f"✅ Pexels: {len(items)} sonuç")
    for item in items:
        print(f"   - {item['title'][:50]} ({item['width']}x{item['height']})")
