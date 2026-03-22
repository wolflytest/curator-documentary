"""
Programatik medya seçici — MoneyPrinterTurbo-Extended semantik eşleştirme.

Her sahne için:
  1. Pexels (foto + video) + Wikimedia + YouTube CC'de ara (çoklu keyword)
  2. Her adayı indir  (MD5 önbellek sayesinde tekrar indirilmez)
  3. sentence-transformers + CLIP ile semantik skorla (Gemini Vision fallback)
  4. relevance_score + quality_score toplamına göre sırala → en iyi adayı döndür
"""
import json
import logging
from pathlib import Path

from documentary_system.state.documentary_state import DocumentaryState, SceneState

log = logging.getLogger(__name__)

_MAX_CANDIDATES = 8     # İndirilecek toplam aday limiti


def select_best_media(scene: SceneState, state: DocumentaryState) -> dict | None:
    """
    Sahne için semantik olarak en uygun medyayı seç.

    Returns:
        dict ile alan: local_path, source, source_url, media_type,
        relevance_score, quality_score, dominant_colors, clip_start
        Ya da None (aday bulunamazsa)
    """
    candidates = _gather_candidates(scene, state.used_media_hashes)
    if not candidates:
        log.warning("[Scene %d] Hiç aday medya bulunamadı.", scene.index)
        return None

    scored = _score_candidates(candidates, scene)
    if not scored:
        log.warning("[Scene %d] Tüm adaylar reddedildi.", scene.index)
        return None

    scored.sort(key=lambda x: x["relevance_score"] + x["quality_score"], reverse=True)
    best = scored[0]
    log.info(
        "[Scene %d] En iyi medya: %s  rel=%.1f  qual=%.1f  kaynak=%s",
        scene.index,
        Path(best["local_path"]).name,
        best["relevance_score"],
        best["quality_score"],
        best["source"],
    )
    return best


# ── Arama + İndirme ───────────────────────────────────────────────────────────

def _gather_candidates(scene: SceneState, used_hashes: set) -> list[dict]:
    """Birden fazla kaynaktan aday topla ve hepsini indir."""
    from documentary_system.tools.media_download_tool import MediaDownloadTool
    from documentary_system.tools.pexels_tool import PexelsTool
    from documentary_system.tools.wikimedia_tool import WikimediaTool
    from documentary_system.tools.ytcc_tool import YouTubeCCTool

    pexels  = PexelsTool()
    wiki    = WikimediaTool()
    ytcc    = YouTubeCCTool()
    dl      = MediaDownloadTool()
    from documentary_system.tools.pixabay_tool import PixabayTool
    pixabay = PixabayTool()

    raw: list[dict] = []

    for kw in (scene.search_keywords or [])[:3]:
        # Pexels fotoğraf
        try:
            items = json.loads(pexels._run(json.dumps({"keyword": kw, "type": "photo", "limit": 3})))
            raw.extend([{**i, "source": "pexels", "media_type": "photo"} for i in items])
        except Exception as exc:
            log.debug("Pexels foto hata (%s): %s", kw, exc)

        # Pexels video
        try:
            items = json.loads(pexels._run(json.dumps({"keyword": kw, "type": "video", "limit": 2})))
            raw.extend([{**i, "source": "pexels", "media_type": "video"} for i in items])
        except Exception as exc:
            log.debug("Pexels video hata (%s): %s", kw, exc)

        # Pixabay video (MPT-Extended kaynağı)
        try:
            items = json.loads(pixabay._run(json.dumps({"keyword": kw, "limit": 2})))
            raw.extend(items)
        except Exception as exc:
            log.debug("Pixabay hata (%s): %s", kw, exc)

        # Wikimedia
        try:
            items = json.loads(wiki._run(json.dumps({"keyword": kw, "limit": 3})))
            raw.extend([{**i, "source": "wikimedia", "media_type": "photo"} for i in items])
        except Exception as exc:
            log.debug("Wikimedia hata (%s): %s", kw, exc)

    # YouTube CC — sadece ilk keyword, 1 video
    if scene.search_keywords:
        try:
            items = json.loads(ytcc._run(json.dumps({"keyword": scene.search_keywords[0], "limit": 1})))
            raw.extend([{**i, "source": "ytcc", "media_type": "video"} for i in items])
        except Exception as exc:
            log.debug("YouTube CC hata: %s", exc)

    # Duplicate URL ve kullanılmış hash temizle
    seen: set = set()
    unique: list[dict] = []
    for c in raw:
        url = c.get("download_url") or c.get("url", "") or c.get("local_path", "")
        key = url[:64]
        if key and key not in seen and key not in used_hashes:
            seen.add(key)
            unique.append(c)

    # İndir
    candidates: list[dict] = []
    for c in unique[:_MAX_CANDIDATES]:
        local_path = c.get("local_path", "")  # ytcc zaten indirmiş olabilir

        if not local_path:
            download_url = c.get("download_url") or c.get("url", "")
            if not download_url:
                continue
            result = json.loads(dl._run(json.dumps({
                "url": download_url,
                "media_type": c.get("media_type", "photo"),
            })))
            if result.get("success"):
                local_path = result["local_path"]

        if local_path and Path(local_path).exists():
            candidates.append({
                "local_path":  local_path,
                "source":      c.get("source", "pexels"),
                "source_url":  c.get("url", ""),
                "media_type":  c.get("media_type", "photo"),
                "duration":    c.get("duration", 0),
            })

    log.info("[Scene %d] %d aday indirildi", scene.index, len(candidates))
    return candidates


# ── Semantik Skorlama (sentence-transformers + CLIP) ──────────────────────────

def _score_candidates(candidates: list[dict], scene: SceneState) -> list[dict]:
    """
    sentence-transformers + CLIP ile semantik skorla.
    Gemini Vision'dan çok daha hızlı, rate limit yok, API maliyeti yok.
    """
    try:
        from documentary_system.services.semantic_matcher import get_matcher
        matcher = get_matcher()
        return matcher.rank_candidates(
            candidates=candidates,
            narration=scene.narration,
            keywords=scene.search_keywords or [],
            visual_description=scene.visual_description or "",
        )
    except Exception as exc:
        log.warning("[Scene %d] Semantik skorlama hatası: %s — Gemini Vision fallback", scene.index, exc)
        return _score_candidates_gemini_fallback(candidates, scene)


def _score_candidates_gemini_fallback(candidates: list[dict], scene: SceneState) -> list[dict]:
    """Gemini Vision fallback — semantic_matcher kullanılamadığında."""
    from documentary_system.tools.gemini_vision_tool import GeminiVisionTool

    vision = GeminiVisionTool()
    scored: list[dict] = []
    vision_calls = 0
    _MAX_VISION_CALLS = 5

    for c in candidates:
        media_type = c.get("media_type", "photo")

        if media_type == "video":
            vid_dur = float(c.get("duration", 0))
            rel = 6.5 if vid_dur >= scene.duration_sec else 5.0
            scored.append({**c, "relevance_score": rel, "quality_score": 7.0,
                           "dominant_colors": [], "clip_start": 0.0})
            continue

        if vision_calls >= _MAX_VISION_CALLS:
            scored.append({**c, "relevance_score": 4.5, "quality_score": 5.0,
                           "dominant_colors": [], "clip_start": 0.0})
            continue

        try:
            raw = vision._run(json.dumps({
                "image_path":     c["local_path"],
                "scene_text":     scene.narration,
                "keywords":       scene.search_keywords,
                "visual_context": scene.visual_description,
            }))
            result = json.loads(raw)
            vision_calls += 1

            if result.get("kullanilabilir", False) and not result.get("watermark_var", False):
                scored.append({
                    **c,
                    "relevance_score": float(result.get("alakalilik_skoru", 5)),
                    "quality_score":   float(result.get("kalite_skoru", 5)),
                    "dominant_colors": result.get("dominant_renkler", []),
                    "clip_start":      0.0,
                })
            else:
                log.debug("[Scene %d] Reddedildi: %s — %s",
                          scene.index, Path(c["local_path"]).name,
                          result.get("red_nedeni", "bilinmiyor"))
        except Exception as exc:
            log.warning("[Scene %d] Vision skoru hatası: %s", scene.index, exc)

    return scored
