"""
TTS Servisi — MoneyPrinterTurbo-Extended tarzı.

İngilizce: edge_tts (Microsoft, ücretsiz, kelime bazlı zaman damgaları)
Türkçe:    KokoroTTS (ONNX) → başarısız → gTTS
           faster-whisper ile kelime zaman damgaları hizalama

Word timestamps formatı:
    [{"word": "Ottoman", "start": 0.12, "end": 0.54}, ...]
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class TTSResult:
    success: bool
    audio_path: str = ""
    duration_sec: float = 0.0
    word_timestamps: list[dict] = field(default_factory=list)
    error: str = ""


# ── English TTS: edge_tts ─────────────────────────────────────────────────────

async def _edge_tts_async(text: str, voice: str, output_path: Path) -> TTSResult:
    """
    edge_tts ile ses üret.
    edge_tts 7.x yalnızca SentenceBoundary verir; kelime hizalaması
    daha sonra faster-whisper ile yapılır.
    """
    try:
        import edge_tts

        communicate = edge_tts.Communicate(text, voice)
        audio_data  = bytearray()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])

        output_path.write_bytes(bytes(audio_data))
        duration_sec = _probe_duration(output_path)

        log.info("edge_tts tamamlandı: %s (%.1fs)", output_path.name, duration_sec)
        return TTSResult(
            success=True,
            audio_path=str(output_path),
            duration_sec=duration_sec,
            # word_timestamps boş — synthesize() içinde faster-whisper ile doldurulur
            word_timestamps=[],
        )

    except Exception as exc:
        log.error("edge_tts hatası: %s", exc)
        return TTSResult(success=False, error=str(exc))


def _run_edge_tts(text: str, voice: str, output_path: Path) -> TTSResult:
    """Sync wrapper for edge_tts."""
    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(_edge_tts_async(text, voice, output_path))
        loop.close()
        return result
    except Exception as exc:
        return TTSResult(success=False, error=str(exc))


# ── Turkish TTS: Kokoro → gTTS + faster-whisper ───────────────────────────────

def _kokoro_tts(text: str, output_path: Path, voice: str = "bf_emma") -> TTSResult:
    """Kokoro ONNX TTS ile ses üret."""
    try:
        from documentary_system.tools.kokoro_tts_tool import KokoroTTSTool
        import json

        tool   = KokoroTTSTool()
        result = json.loads(tool._run(json.dumps({
            "text":        text,
            "output_path": str(output_path),
            "voice":       voice,
            "language":    "en",
            "speed":       0.95,
        })))
        if result.get("success"):
            return TTSResult(
                success=True,
                audio_path=result["output_path"],
                duration_sec=result.get("duration_sec", 0.0),
            )
        return TTSResult(success=False, error="kokoro returned failure")
    except Exception as exc:
        return TTSResult(success=False, error=str(exc))


def _gtts_tts(text: str, output_path: Path, lang: str = "tr") -> TTSResult:
    """gTTS fallback."""
    try:
        from gtts import gTTS
        from pydub import AudioSegment

        tmp_mp3 = output_path.with_suffix(".gtts_tmp.mp3")
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(str(tmp_mp3))

        # Standardize to 16kHz mono (faster-whisper uyumu için)
        audio = AudioSegment.from_mp3(str(tmp_mp3)).set_channels(1).set_frame_rate(16000)
        audio.export(str(output_path), format="mp3")
        tmp_mp3.unlink(missing_ok=True)

        duration_sec = len(audio) / 1000.0
        log.info("gTTS tamamlandı: %s (%.1fs)", output_path.name, duration_sec)
        return TTSResult(success=True, audio_path=str(output_path), duration_sec=duration_sec)
    except Exception as exc:
        log.error("gTTS hatası: %s", exc)
        return TTSResult(success=False, error=str(exc))


def _align_with_whisper(audio_path: Path, text: str) -> list[dict]:
    """faster-whisper ile kelime zaman damgaları üret."""
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            initial_prompt=text[:200],
        )
        words = []
        for seg in segments:
            for w in (seg.words or []):
                words.append({
                    "word":  w.word.strip(),
                    "start": round(float(w.start), 3),
                    "end":   round(float(w.end), 3),
                })
        log.info("faster-whisper hizalama: %d kelime", len(words))
        return words
    except Exception as exc:
        log.warning("faster-whisper hizalama başarısız: %s", exc)
        return []


# ── Yardımcı ─────────────────────────────────────────────────────────────────

def _probe_duration(path: Path) -> float:
    import subprocess
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


# ── Public API ────────────────────────────────────────────────────────────────

def synthesize(
    text: str,
    output_path: Path,
    language: str = "tr",
    voice: str = "gtts_tr",
    align_words: bool = True,
) -> TTSResult:
    """
    Metinden ses üret + isteğe bağlı kelime zaman damgaları.

    Args:
        text:         Anlatım metni
        output_path:  Çıktı ses dosyası (.mp3)
        language:     'tr' | 'en'
        voice:        edge_tts ses kodu (en için) veya 'gtts_tr' (tr için)
        align_words:  True → faster-whisper ile kelime hizalaması

    Returns:
        TTSResult (success, audio_path, duration_sec, word_timestamps)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if language == "en":
        result = _run_edge_tts(text, voice, output_path)
        if not result.success:
            log.warning("edge_tts başarısız, Kokoro'ya geçiliyor: %s", result.error)
            result = _kokoro_tts(text, output_path, voice)
    else:
        # Türkçe: Kokoro → gTTS
        result = _kokoro_tts(text, output_path, voice="bf_emma")
        if not result.success:
            log.warning("Kokoro başarısız, gTTS'e geçiliyor.")
            result = _gtts_tts(text, output_path, lang="tr")

    # faster-whisper ile kelime hizalaması (EN + TR)
    if result.success and align_words and not result.word_timestamps:
        result.word_timestamps = _align_with_whisper(Path(result.audio_path), text)

    return result
