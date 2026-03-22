"""
Pixabay video arama aracı — MPT-Extended material.py'den adapte edildi.
Ücretsiz lisanslı stok videolar (Pixabay License).
"""
import json
import logging
import os
from urllib.parse import urlencode

import requests
from crewai.tools import BaseTool

log = logging.getLogger(__name__)

_PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "")


class PixabayTool(BaseTool):
    name: str = "Pixabay Video Ara"
    description: str = """
    Pixabay'de ücretsiz stok video ara.
    Input: JSON string {"keyword": "...", "limit": 3}
    Output: JSON list [{"url": str, "duration": float, "source": "pixabay"}]
    """

    def _run(self, input_str: str) -> str:
        try:
            data = json.loads(input_str)
            keyword = data.get("keyword", "")
            limit = int(data.get("limit", 3))
        except Exception:
            keyword = input_str.strip()
            limit = 3

        if not _PIXABAY_API_KEY:
            log.debug("PIXABAY_API_KEY ayarlanmamış, atlanıyor.")
            return json.dumps([])

        if not keyword:
            return json.dumps([])

        try:
            params = {
                "q":          keyword,
                "video_type": "all",
                "per_page":   min(limit * 3, 20),
                "key":        _PIXABAY_API_KEY,
            }
            url = f"https://pixabay.com/api/videos/?{urlencode(params)}"
            r = requests.get(url, timeout=(10, 30))
            r.raise_for_status()
            hits = r.json().get("hits", [])

            results = []
            for v in hits[:limit * 2]:
                duration = v.get("duration", 0)
                videos   = v.get("videos", {})
                # En iyi kaliteyi seç: large → medium → small → tiny
                for quality in ("large", "medium", "small", "tiny"):
                    vf = videos.get(quality, {})
                    if vf.get("url"):
                        results.append({
                            "url":          vf["url"],
                            "download_url": vf["url"],
                            "duration":     duration,
                            "source":       "pixabay",
                            "media_type":   "video",
                        })
                        break
                if len(results) >= limit:
                    break

            log.info("Pixabay '%s': %d sonuç", keyword, len(results))
            return json.dumps(results)

        except Exception as exc:
            log.warning("Pixabay arama hatası (%s): %s", keyword, exc)
            return json.dumps([])


if __name__ == "__main__":
    tool = PixabayTool()
    print("✅ PixabayTool yüklendi")
    print("   Not: PIXABAY_API_KEY gereklidir")
