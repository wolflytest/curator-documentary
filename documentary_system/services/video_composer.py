"""
Video birleştirici — MPT-Extended video_effects.py + curator özellikleri.

MPT-Extended'dan alınan geçişler (orijinal kod):
  fadein_transition, fadeout_transition, slidein_transition, slideout_transition

Curator eklentileri:
  - Ken Burns (FFmpegTool üzerinden)
  - DiversityTracker
  - Türkçe TTS uyumu
  - MoviePy → FFmpeg fallback zinciri

Video kalitesi: MPT-Extended standardı (CRF 18, 8000k bitrate, 30fps)
BGM: 20 MPT-Extended bundled track
"""
from __future__ import annotations

import logging
import random
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

_TRANSITION_DURATION = 0.5  # saniye
_BG_MUSIC_VOLUME     = 0.20  # MPT-Extended default: 20%

# MPT-Extended video kalite ayarları
_VIDEO_CRF     = "18"
_VIDEO_BITRATE = "8000k"
_VIDEO_FPS     = 30
_VIDEO_PRESET  = "medium"

# ── VideoAspect (MPT-Extended schema.py'den) ──────────────────────────────────
class VideoAspect:
    """16:9 (yatay), 9:16 (dikey/portrait), 1:1 (kare)."""
    LANDSCAPE = "16:9"   # 1920x1080
    PORTRAIT  = "9:16"   # 1080x1920
    SQUARE    = "1:1"    # 1080x1080

    @staticmethod
    def to_resolution(aspect: str) -> tuple[int, int]:
        if aspect == "9:16":
            return 1080, 1920
        elif aspect == "1:1":
            return 1080, 1080
        return 1920, 1080  # 16:9 default


def _apply_transition(clip, transition: str, duration: float = _TRANSITION_DURATION):
    """
    MPT-Extended video_effects.py geçişlerini uygula.
    Desteklenen: fade, fadein, fadeout, slidein_left, slidein_right,
                 slidein_top, slidein_bottom, slideout_left, slideout_right,
                 slideout_top, slideout_bottom, shuffle (rastgele), cut
    """
    from documentary_system.services.video_effects import (
        fadein_transition, fadeout_transition,
        slidein_transition, slideout_transition,
    )
    t = transition.lower()
    if t in ("fade", "fadein"):
        clip = fadein_transition(clip, duration)
    if t in ("fade", "fadeout"):
        clip = fadeout_transition(clip, duration)
    elif t.startswith("slidein_"):
        side = t.replace("slidein_", "") or "left"
        clip = slidein_transition(clip, duration, side)
    elif t.startswith("slideout_"):
        side = t.replace("slideout_", "") or "right"
        clip = slideout_transition(clip, duration, side)
    elif t == "shuffle":
        # Rastgele bir geçiş seç
        options = ["fadein", "slidein_left", "slidein_right", "slidein_top", "slidein_bottom"]
        clip = _apply_transition(clip, random.choice(options), duration)
    # "cut" veya bilinmeyen → geçiş yok
    return clip


def compose_scene(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    duration: float,
    transition: str = "cut",
    aspect: str = "16:9",
) -> Path | None:
    """
    Video klibine ses ekle, süreyi kırp, MPT-Extended geçişlerini uygula.
    Video kalitesi: CRF 18, 8000k bitrate, 30fps (MPT-Extended standardı)
    aspect: "16:9" | "9:16" | "1:1"
    """
    try:
        from moviepy import VideoFileClip, AudioFileClip, ColorClip, CompositeVideoClip

        video = VideoFileClip(str(video_path)).subclipped(0, duration)

        # Aspect ratio yeniden boyutlandırma (MPT-Extended'dan)
        target_w, target_h = VideoAspect.to_resolution(aspect)
        vw, vh = video.size
        if (vw, vh) != (target_w, target_h):
            clip_ratio   = vw / vh
            target_ratio = target_w / target_h
            if abs(clip_ratio - target_ratio) > 0.01:
                scale = target_w / vw if clip_ratio > target_ratio else target_h / vh
                nw, nh = int(vw * scale), int(vh * scale)
                bg    = ColorClip(size=(target_w, target_h), color=(0, 0, 0)).with_duration(duration)
                video = CompositeVideoClip([bg, video.resized((nw, nh)).with_position("center")])
            else:
                video = video.resized((target_w, target_h))

        audio = AudioFileClip(str(audio_path))
        video = video.with_audio(audio.subclipped(0, min(audio.duration, duration)))

        if transition != "cut":
            video = _apply_transition(video, transition)

        video.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=_VIDEO_FPS,
            preset=_VIDEO_PRESET,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", _VIDEO_CRF, "-b:v", _VIDEO_BITRATE,
                           "-movflags", "+faststart"],
            logger=None,
        )
        log.info("compose_scene tamamlandı: %s (%.1fs, geçiş=%s, aspect=%s)",
                 output_path.name, duration, transition, aspect)
        return output_path

    except Exception as exc:
        log.warning("MoviePy compose_scene hatası: %s — FFmpeg'e geçiliyor", exc)
        return _ffmpeg_combine(video_path, audio_path, output_path, duration)


def _ffmpeg_combine(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    duration: float,
) -> Path | None:
    """FFmpeg ile video+ses birleştir (MoviePy fallback)."""
    try:
        subprocess.run(
            ["ffmpeg", "-y",
             "-i", str(video_path),
             "-i", str(audio_path),
             "-map", "0:v:0",
             "-map", "1:a:0",
             "-c:v", "copy",
             "-c:a", "aac",
             "-b:a", "128k",
             "-t", str(duration),
             "-shortest",
             str(output_path)],
            check=True, capture_output=True,
        )
        return output_path
    except subprocess.CalledProcessError as exc:
        log.error("FFmpeg combine hatası: %s", exc.stderr.decode(errors="replace")[:300])
        return None


def concat_clips(
    clip_paths: list[Path],
    output_path: Path,
) -> Path:
    """
    Klipleri birleştir. MoviePy → FFmpeg fallback.
    """
    try:
        from moviepy import VideoFileClip, concatenate_videoclips

        clips = [VideoFileClip(str(p)) for p in clip_paths]
        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=_VIDEO_FPS,
            preset=_VIDEO_PRESET,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", _VIDEO_CRF, "-b:v", _VIDEO_BITRATE, "-movflags", "+faststart"],
            logger=None,
        )
        for c in clips:
            c.close()
        log.info("Klipler birleştirildi: %s (%d klip)", output_path.name, len(clip_paths))
        return output_path

    except Exception as exc:
        log.warning("MoviePy concat hatası: %s — FFmpeg concat'e geçiliyor", exc)
        return _ffmpeg_concat(clip_paths, output_path)


def _ffmpeg_concat(clip_paths: list[Path], output_path: Path) -> Path:
    """FFmpeg concat demuxer ile birleştir."""
    concat_file = output_path.parent / "concat_list.txt"
    concat_file.write_text(
        "\n".join(f"file '{p}'" for p in clip_paths),
        encoding="utf-8",
    )
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "concat",
         "-safe", "0",
         "-i", str(concat_file),
         "-c:v", "copy",
         "-c:a", "aac",
         "-b:a", "128k",
         "-movflags", "+faststart",
         str(output_path)],
        check=True, capture_output=True,
    )
    return output_path


def add_background_music(
    video_path: Path,
    music_path: Path,
    output_path: Path,
    volume: float = _BG_MUSIC_VOLUME,
) -> Path | None:
    """
    Arka plan müziği ekle — MoviePy AudioLoop + MultiplyVolume.
    FFmpeg'e fallback mevcut.
    """
    try:
        from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
        from moviepy.audio.fx import AudioLoop, MultiplyVolume

        video  = VideoFileClip(str(video_path))
        music  = AudioFileClip(str(music_path))
        music  = AudioLoop(duration=video.duration).apply(music)
        music  = MultiplyVolume(volume).apply(music)

        mixed  = CompositeAudioClip([video.audio, music])
        final  = video.with_audio(mixed)
        final.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=_VIDEO_FPS,
            preset=_VIDEO_PRESET,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", _VIDEO_CRF, "-b:v", _VIDEO_BITRATE, "-movflags", "+faststart"],
            logger=None,
        )
        video.close()
        music.close()
        log.info("Arka plan müziği eklendi: %s (vol=%.0f%%)", output_path.name, volume * 100)
        return output_path

    except Exception as exc:
        log.warning("MoviePy BGM hatası: %s — FFmpeg'e geçiliyor", exc)
        return _ffmpeg_bgm(video_path, music_path, output_path, volume)


def _ffmpeg_bgm(
    video_path: Path,
    music_path: Path,
    output_path: Path,
    volume: float,
) -> Path | None:
    """FFmpeg ile arka plan müziği ekle."""
    try:
        subprocess.run(
            ["ffmpeg", "-y",
             "-i", str(video_path),
             "-stream_loop", "-1",
             "-i", str(music_path),
             "-filter_complex",
             f"[1:a]volume={volume},apad[music];[0:a][music]amix=inputs=2:duration=first[aout]",
             "-map", "0:v",
             "-map", "[aout]",
             "-c:v", "copy",
             "-c:a", "aac",
             "-b:a", "192k",
             "-shortest",
             str(output_path)],
            check=True, capture_output=True,
        )
        return output_path
    except subprocess.CalledProcessError as exc:
        log.error("FFmpeg BGM hatası: %s", exc.stderr.decode(errors="replace")[:200])
        return None


# ── Klip çeşitliliği kontrolü ─────────────────────────────────────────────────

class DiversityTracker:
    """
    Aynı medya dosyasının çok fazla kullanılmasını önler.
    MPT-Extended: max_video_reuse=2
    """
    def __init__(self, max_reuse: int = 2):
        self.max_reuse  = max_reuse
        self._counts: dict[str, int] = {}

    def is_allowed(self, local_path: str) -> bool:
        return self._counts.get(local_path, 0) < self.max_reuse

    def register(self, local_path: str) -> None:
        self._counts[local_path] = self._counts.get(local_path, 0) + 1

    def filter_candidates(self, candidates: list[dict]) -> list[dict]:
        """Kullanım limitine ulaşmış adayları filtrele."""
        return [c for c in candidates if self.is_allowed(c.get("local_path", ""))]
