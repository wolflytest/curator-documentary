"""
Belgesel üretim sürecinin merkezi state yönetimi.
SQLite'a serialize edilir — crash sonrası kaldığı yerden devam eder.
"""
import json
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class SceneState:
    """Tek bir belgesel sahnesinin tüm durumu."""

    index: int
    narration: str                     # Türkçe anlatım metni
    search_keywords: list[str]         # İngilizce, medya arama için
    visual_description: str            # Nasıl görsel olmalı
    mood: str                          # dramatic|peaceful|tense|neutral
    transition: str                    # fade|cut|dissolve
    duration_sec: float                # hedef süre (saniye)

    # Arama sonrası dolar
    candidate_media: list[dict] = field(default_factory=list)
    approved_media: dict | None = None

    # Üretim sonrası dolar
    tts_path: str | None = None        # voiceover dosyası
    final_clip_path: str | None = None # işlenmiş video/foto klibi

    # Consistency bilgisi
    color_palette: list[str] = field(default_factory=list)
    era_verified: bool = False

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "narration": self.narration,
            "search_keywords": self.search_keywords,
            "visual_description": self.visual_description,
            "mood": self.mood,
            "transition": self.transition,
            "duration_sec": self.duration_sec,
            "candidate_media": self.candidate_media,
            "approved_media": self.approved_media,
            "tts_path": self.tts_path,
            "final_clip_path": self.final_clip_path,
            "color_palette": self.color_palette,
            "era_verified": self.era_verified,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SceneState":
        return cls(
            index=d["index"],
            narration=d["narration"],
            search_keywords=d["search_keywords"],
            visual_description=d["visual_description"],
            mood=d.get("mood", "neutral"),
            transition=d.get("transition", "fade"),
            duration_sec=float(d.get("duration_sec", 7.0)),
            candidate_media=d.get("candidate_media", []),
            approved_media=d.get("approved_media"),
            tts_path=d.get("tts_path"),
            final_clip_path=d.get("final_clip_path"),
            color_palette=d.get("color_palette", []),
            era_verified=d.get("era_verified", False),
        )


@dataclass
class DocumentaryState:
    """Belgeselin tüm üretim sürecini izleyen merkezi state."""

    doc_id: int
    topic: str
    status: str = "pending"  # pending|scripting|searching|assembling|qa|done|error

    # Script crew çıktısı
    title: str | None = None
    description: str | None = None      # YouTube için
    tags: list[str] = field(default_factory=list)
    scenes: list[SceneState] = field(default_factory=list)

    # Consistency tracking
    visual_style: dict = field(default_factory=dict)
    used_media_hashes: set[str] = field(default_factory=set)

    # Feedback loop
    script_revision_count: int = 0
    script_critic_notes: str = ""

    # Sonuç
    output_path: str | None = None
    youtube_url: str | None = None
    error_msg: str | None = None

    def to_json(self) -> str:
        """State'i JSON string'e serialize et."""
        return json.dumps(
            {
                "doc_id": self.doc_id,
                "topic": self.topic,
                "status": self.status,
                "title": self.title,
                "description": self.description,
                "tags": self.tags,
                "scenes": [s.to_dict() for s in self.scenes],
                "visual_style": self.visual_style,
                "used_media_hashes": list(self.used_media_hashes),
                "script_revision_count": self.script_revision_count,
                "script_critic_notes": self.script_critic_notes,
                "output_path": self.output_path,
                "youtube_url": self.youtube_url,
                "error_msg": self.error_msg,
            },
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "DocumentaryState":
        """JSON string'den DocumentaryState oluştur."""
        d = json.loads(json_str)
        state = cls(
            doc_id=d["doc_id"],
            topic=d["topic"],
            status=d.get("status", "pending"),
            title=d.get("title"),
            description=d.get("description"),
            tags=d.get("tags", []),
            scenes=[SceneState.from_dict(s) for s in d.get("scenes", [])],
            visual_style=d.get("visual_style", {}),
            used_media_hashes=set(d.get("used_media_hashes", [])),
            script_revision_count=d.get("script_revision_count", 0),
            script_critic_notes=d.get("script_critic_notes", ""),
            output_path=d.get("output_path"),
            youtube_url=d.get("youtube_url"),
            error_msg=d.get("error_msg"),
        )
        return state

    def scene_completion_ratio(self) -> float:
        """Tamamlanan sahne oranını döndür (0.0 - 1.0)."""
        if not self.scenes:
            return 0.0
        done = sum(1 for s in self.scenes if s.final_clip_path is not None)
        return done / len(self.scenes)

    def get_visual_context_summary(self) -> str:
        """
        Son 3 sahnedeki görsel bilgileri özetle.
        ConsistencyAgent bunu context olarak kullanır.
        """
        completed = [s for s in self.scenes if s.approved_media is not None]
        recent = completed[-3:] if len(completed) >= 3 else completed
        if not recent:
            return "Henüz onaylanmış sahne yok — görsel stil serbest."

        lines = ["Son onaylanan sahneler:"]
        for s in recent:
            media = s.approved_media or {}
            lines.append(
                f"  Sahne {s.index}: '{s.narration[:60]}...'\n"
                f"    Kaynak: {media.get('source', '?')}, Tip: {media.get('media_type', '?')}\n"
                f"    Renkler: {', '.join(s.color_palette) or 'belirsiz'}"
            )
        if self.visual_style:
            lines.append(f"\nGenel görsel stil: {self.visual_style}")
        return "\n".join(lines)


if __name__ == "__main__":
    scene = SceneState(
        index=0,
        narration="Osmanlı İmparatorluğu'nun kuruluşu...",
        search_keywords=["ottoman empire founding 1299"],
        visual_description="Tarihi harita, savaş sahnesi",
        mood="dramatic",
        transition="fade",
        duration_sec=8.0,
    )
    state = DocumentaryState(doc_id=1, topic="Osmanlı'nın Yükselişi")
    state.scenes.append(scene)
    state.title = "Osmanlı'nın Yükselişi: 600 Yıllık Bir Efsane"
    serialized = state.to_json()
    restored = DocumentaryState.from_json(serialized)
    assert restored.doc_id == state.doc_id
    assert restored.scenes[0].narration == scene.narration
    print("✅ DocumentaryState serializasyon testi OK")
    print(f"   Tamamlanma oranı: {state.scene_completion_ratio():.0%}")
