"""
Semantik medya eşleştirme servisi — MoneyPrinterTurbo-Extended tarzı.

İki model birlikte çalışır:
  1. sentence-transformers (all-mpnet-base-v2)  → metin-metin benzerliği
  2. CLIP (clip-ViT-B-32)                       → görsel-metin benzerliği

Kullanım:
    from documentary_system.services.semantic_matcher import SemanticMatcher
    matcher = SemanticMatcher()
    score = matcher.text_similarity("Ottoman army", ["ottoman", "empire", "battle"])
    score = matcher.image_text_score("/tmp/img.jpg", "Ottoman army marching")
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

# Model yükleme lazy — ilk çağrıda gerçekleşir
_st_model = None
_clip_model = None


def _load_st():
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer("all-mpnet-base-v2")
            log.info("SentenceTransformer yüklendi: all-mpnet-base-v2")
        except Exception as exc:
            log.warning("SentenceTransformer yüklenemedi: %s", exc)
    return _st_model


def _load_clip():
    global _clip_model
    if _clip_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _clip_model = SentenceTransformer("clip-ViT-B-32")
            log.info("CLIP modeli yüklendi: clip-ViT-B-32")
        except Exception as exc:
            log.warning("CLIP modeli yüklenemedi: %s", exc)
    return _clip_model


class SemanticMatcher:
    """
    Metin-metin ve görsel-metin semantik benzerlik skoru hesaplar.
    Tüm skorlar 0.0–10.0 aralığında döner.
    """

    def text_similarity(self, query: str, keywords: list[str]) -> float:
        """
        Sahne narasyonu ile arama keyword'leri arasındaki benzerlik.
        0–10 arası skor döner.
        """
        model = _load_st()
        if model is None or not keywords:
            return 5.0  # fallback: orta skor

        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            kw_text = " ".join(keywords)
            emb = model.encode([query, kw_text], convert_to_numpy=True)
            cos = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
            # cosine [-1,1] → 0-10 ölçeği
            return round((cos + 1) / 2 * 10, 2)
        except Exception as exc:
            log.warning("text_similarity hatası: %s", exc)
            return 5.0

    def image_text_score(self, image_path: str | Path, description: str) -> float:
        """
        CLIP ile görsel ↔ metin benzerliği (MPT-Extended image_similarity.py kullanır).
        Local dosya yolu veya URL desteklenir.
        0–10 arası skor döner.
        """
        try:
            from documentary_system.services.image_similarity import (
                calculate_text_image_similarity,
                IMAGE_SIMILARITY_AVAILABLE,
            )
            if not IMAGE_SIMILARITY_AVAILABLE:
                raise ImportError("CLIP dependencies not available")

            path_str = str(image_path)
            # Local dosya → file:// benzeri bir URL olmadığından
            # MPT-Extended'ın URL-based fonksiyonu yerine local file load kullanıyoruz
            if Path(path_str).exists():
                # Local dosyayı MPT-Extended benzeri şekilde işle
                return self._image_text_score_local(path_str, description)
            else:
                # URL varsayımı — MPT-Extended fonksiyonunu doğrudan kullan
                raw = calculate_text_image_similarity(description, path_str)
                return round(raw * 10, 2)
        except Exception as exc:
            log.warning("image_text_score MPT hatası: %s, sentence-transformers fallback", exc)
            return self._image_text_score_st_fallback(image_path, description)

    def _image_text_score_local(self, image_path: str, description: str) -> float:
        """MPT-Extended embedding cache + CLIPModel ile local görsel skoru."""
        try:
            import gc
            import torch
            from transformers import CLIPProcessor, CLIPModel
            from PIL import Image

            # Singleton cache
            if not hasattr(self, "_clip_model"):
                self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
                self._clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
                self._img_cache: dict = {}
                self._txt_cache: dict = {}

            # Cache kontrolü
            if image_path in self._img_cache and description in self._txt_cache:
                img_emb  = self._img_cache[image_path]
                txt_emb  = self._txt_cache[description]
                sim = torch.cosine_similarity(txt_emb, img_emb, dim=-1).item()
                return round(((sim + 1) / 2) * 10, 2)

            img = Image.open(image_path).convert("RGB")
            if max(img.size) > 512:
                img.thumbnail((512, 512), Image.Resampling.LANCZOS)

            inputs = self._clip_proc(text=[description], images=img, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                img_emb = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
                txt_emb = outputs.text_embeds  / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)

            # Cache'e ekle (max 100)
            if len(self._img_cache) < 100:
                self._img_cache[image_path] = img_emb.clone()
            if len(self._txt_cache) < 100:
                self._txt_cache[description] = txt_emb.clone()

            sim = torch.cosine_similarity(txt_emb, img_emb, dim=-1).item()
            gc.collect()
            return round(((sim + 1) / 2) * 10, 2)
        except Exception as exc:
            log.warning("_image_text_score_local hatası: %s", exc)
            return 5.0

    def _image_text_score_st_fallback(self, image_path: str | Path, description: str) -> float:
        """sentence-transformers CLIP fallback."""
        model = _load_clip()
        if model is None:
            return 5.0
        try:
            from PIL import Image
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            img = Image.open(str(image_path)).convert("RGB")
            img_emb  = model.encode(img, convert_to_numpy=True)
            text_emb = model.encode(description, convert_to_numpy=True)
            cos = float(cosine_similarity([img_emb], [text_emb])[0][0])
            return round((cos + 1) / 2 * 10, 2)
        except Exception as exc:
            log.warning("_image_text_score_st_fallback hatası: %s", exc)
            return 5.0

    def rank_candidates(
        self,
        candidates: list[dict],
        narration: str,
        keywords: list[str],
        visual_description: str,
    ) -> list[dict]:
        """
        Aday listesini semantik skorlara göre sırala.

        Her aday dict'e şu alanlar eklenir:
            relevance_score  (0–10)
            quality_score    (0–10, sabit 7.0 — görsel kalite için)
            dominant_colors  ([])
            clip_start       (0.0)

        Sıralama: relevance_score + quality_score (desc)
        """
        scored = []
        query = f"{narration} {visual_description}"

        for c in candidates:
            media_type = c.get("media_type", "photo")

            if media_type == "video":
                # Video: sadece süre uyumu skoru
                vid_dur = float(c.get("duration", 0))
                # Süre skoru dışında metin-keyword uyumu da ekle
                kw_score = self.text_similarity(query, keywords)
                rel = (6.5 if vid_dur > 0 else 5.0) * 0.4 + kw_score * 0.6
                scored.append({
                    **c,
                    "relevance_score": round(rel, 2),
                    "quality_score":   7.0,
                    "dominant_colors": [],
                    "clip_start":      0.0,
                })
                continue

            # Fotoğraf: CLIP + metin benzerliği
            local_path = c.get("local_path", "")
            if not local_path or not Path(local_path).exists():
                continue

            text_score  = self.text_similarity(query, keywords)
            clip_score  = self.image_text_score(local_path, query)
            # Ağırlıklı ortalama: CLIP %60, metin %40
            rel = round(clip_score * 0.6 + text_score * 0.4, 2)

            scored.append({
                **c,
                "relevance_score": rel,
                "quality_score":   7.0,
                "dominant_colors": [],
                "clip_start":      0.0,
            })

        scored.sort(key=lambda x: x["relevance_score"] + x["quality_score"], reverse=True)
        return scored


# Module-level singleton
_matcher: SemanticMatcher | None = None


def get_matcher() -> SemanticMatcher:
    global _matcher
    if _matcher is None:
        _matcher = SemanticMatcher()
    return _matcher
