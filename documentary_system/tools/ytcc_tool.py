"""
YouTube Creative Commons lisanslı video arama ve indirme aracı.
Cookie dosyası ile bot tespitini aşar.
"""
import json
import logging
from pathlib import Path

import yt_dlp
from crewai.tools import BaseTool

log = logging.getLogger(__name__)

_TMP_DIR = Path("/tmp/curator_docs/ytcc")
_TMP_DIR.mkdir(parents=True, exist_ok=True)

COOKIE_FILE = Path("/home/ubuntu/curator-documentary/youtube_cookies.txt")


class YouTubeCCTool(BaseTool):
    name: str = "YouTube Creative Commons Video Ara"
    description: str = """
    YouTube'da Creative Commons lisanslı video arar ve indirir.
    Input: JSON string {"keyword": "ottoman empire documentary", "limit": 3}
    Output: JSON list of {title, url, local_path, duration}
    """

    def _run(self, input_str: str) -> str:
        try:
            try:
                data = json.loads(input_str)
            except (json.JSONDecodeError, ValueError):
                data = {"keyword": input_str.strip(), "limit": 3}

            keyword = data.get("keyword", "")
            limit = min(int(data.get("limit", 3)), 5)
        except Exception as exc:
            log.error("YouTubeCCTool geçersiz input: %s", exc)
            return json.dumps([])

        if not keyword:
            return json.dumps([])

        def cc_filter(info: dict) -> str | None:
            license_val = info.get("license", "") or ""
            if "Creative Commons" not in license_val:
                return "CC lisanslı değil"
            duration = info.get("duration", 0) or 0
            if duration > 300:
                return "5 dakikadan uzun"
            return None

        ydl_opts = {
            "match_filter": cc_filter,
            "format": "best[height<=720]",
            "outtmpl": str(_TMP_DIR / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "playlist_items": f"1-{limit}",
        }

        if COOKIE_FILE.exists():
            ydl_opts["cookiefile"] = str(COOKIE_FILE)
            log.info("YouTube cookie dosyası kullanılıyor: %s", COOKIE_FILE)
        else:
            log.warning("YouTube cookie dosyası bulunamadı: %s", COOKIE_FILE)

        results = []
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_url = f"ytsearch{limit}:{keyword} documentary creative commons"
                info = ydl.extract_info(search_url, download=True)
                entries = info.get("entries", []) if info else []
                for entry in entries:
                    if entry is None:
                        continue
                    video_id = entry.get("id", "")
                    ext = entry.get("ext", "mp4")
                    local_path = _TMP_DIR / f"{video_id}.{ext}"
                    if local_path.exists() and local_path.stat().st_size > 10000:
                        results.append({
                            "title": entry.get("title", ""),
                            "url": entry.get("webpage_url", ""),
                            "local_path": str(local_path),
                            "duration": entry.get("duration", 0),
                        })
        except Exception as exc:
            log.error("YouTubeCCTool indirme hatası: %s", exc)

        log.info("YouTube CC araması: %d video (%s)", len(results), keyword)
        return json.dumps(results, ensure_ascii=False)


if __name__ == "__main__":
    tool = YouTubeCCTool()
    print(f"Cookie var mı: {COOKIE_FILE.exists()}")
    r = tool._run(json.dumps({"keyword": "ancient rome history", "limit": 1}))
    items = json.loads(r)
    print(f"✅ YouTube CC: {len(items)} sonuç")
