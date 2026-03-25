"""
TTS Servisi — MoneyPrinterTurbo-Extended tarzı.

İngilizce: edge_tts (Microsoft, ücretsiz, kelime bazlı zaman damgaları)
Türkçe:    KokoroTTS (ONNX) → başarısız → gTTS
           faster-whisper ile kelime zaman damgaları hizalama

Ek TTS seçenekleri (MPT-Extended'dan):
  chatterbox:default  — Chatterbox TTS (ses klonlama, whisperx hizalama)
  chatterbox:clone:<name> — Referans ses dosyasıyla ses klonlama
  siliconflow:FunAudioLLM/CosyVoice2-0.5B:<voice> — SiliconFlow CosyVoice2 TTS

Word timestamps formatı:
    [{"word": "Ottoman", "start": 0.12, "end": 0.54}, ...]
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# ── Chatterbox TTS (MPT-Extended) ─────────────────────────────────────────────
try:
    from chatterbox.tts import ChatterboxTTS
    import whisperx
    import torch
    import torchaudio
    CHATTERBOX_AVAILABLE = True
    log.info("Chatterbox TTS ve WhisperX mevcut")
except ImportError as _e:
    CHATTERBOX_AVAILABLE = False
    log.debug("Chatterbox TTS veya WhisperX mevcut değil: %s", _e)

_chatterbox_model = None
_whisperx_model   = None


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


# ── ElevenLabs TTS ────────────────────────────────────────────────────────────

def _elevenlabs_tts(text: str, voice: str, output_path: Path) -> TTSResult:
    """
    ElevenLabs TTS API — yüksek kaliteli neural TTS.
    voice formatı: "elevenlabs:Adam" veya "elevenlabs:<voice_id>"
    ELEVENLABS_API_KEY env değişkeni gereklidir. Ücretsiz: 10.000 karakter/ay.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not api_key:
        return TTSResult(success=False, error="ELEVENLABS_API_KEY ayarlanmamış")

    voice_name = voice.split(":", 1)[1] if ":" in voice else "Adam"

    # Bilinen ses adı → ID eşlemesi
    _voice_ids = {
        "Adam":    "pNInz6obpgDQGcFmaJgB",
        "Antoni":  "ErXwobaYiN019PkySvjV",
        "Arnold":  "VR6AewLTigWG4xSOukaG",
        "Rachel":  "21m00Tcm4TlvDq8ikWAM",
        "Domi":    "AZnzlk1XvdvUeBnXmlld",
    }
    voice_id = _voice_ids.get(voice_name, voice_name)  # doğrudan ID de olabilir

    try:
        import requests as _req
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        resp = _req.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code == 200:
            output_path.write_bytes(resp.content)
            duration = _probe_duration(output_path)
            log.info("ElevenLabs TTS tamamlandı: %s (%.1fs)", output_path.name, duration)
            return TTSResult(success=True, audio_path=str(output_path), duration_sec=duration)
        log.warning("ElevenLabs %d: %s", resp.status_code, resp.text[:200])
        return TTSResult(success=False, error=f"ElevenLabs HTTP {resp.status_code}")
    except Exception as exc:
        return TTSResult(success=False, error=str(exc))


# ── Siliconflow TTS (MPT-Extended) ────────────────────────────────────────────

def _siliconflow_tts(text: str, voice: str, output_path: Path) -> TTSResult:
    """
    SiliconFlow CosyVoice2 TTS API.
    voice formatı: "siliconflow:FunAudioLLM/CosyVoice2-0.5B:alex-Male"
    SILICONFLOW_API_KEY env değişkeni gereklidir.
    """
    api_key = os.getenv("SILICONFLOW_API_KEY", "")
    if not api_key:
        return TTSResult(success=False, error="SILICONFLOW_API_KEY ayarlanmamış")

    parts = voice.split(":")
    if len(parts) < 3:
        return TTSResult(success=False, error=f"Geçersiz siliconflow ses formatı: {voice}")

    model     = parts[1]
    voice_str = parts[2].split("-")[0]  # "alex-Male" → "alex"
    full_voice = f"{model}:{voice_str}"

    try:
        import requests as _req
        payload = {
            "model":           model,
            "input":           text.strip(),
            "voice":           full_voice,
            "response_format": "mp3",
            "sample_rate":     32000,
            "stream":          False,
            "speed":           1.0,
            "gain":            0,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        for attempt in range(3):
            try:
                resp = _req.post(
                    "https://api.siliconflow.cn/v1/audio/speech",
                    json=payload, headers=headers, timeout=60,
                )
                if resp.status_code == 200:
                    output_path.write_bytes(resp.content)
                    duration = _probe_duration(output_path)
                    log.info("SiliconFlow TTS tamamlandı: %s (%.1fs)", output_path.name, duration)
                    return TTSResult(success=True, audio_path=str(output_path), duration_sec=duration)
                log.warning("SiliconFlow %d: %s", resp.status_code, resp.text[:200])
            except Exception as exc:
                log.warning("SiliconFlow deneme %d hatası: %s", attempt + 1, exc)

        return TTSResult(success=False, error="SiliconFlow 3 denemede başarısız")
    except Exception as exc:
        return TTSResult(success=False, error=str(exc))


# ── Chatterbox TTS (MPT-Extended) ─────────────────────────────────────────────

def _preprocess_text_chatterbox(text: str) -> str:
    """Chatterbox TTS için metin ön işleme (MPT-Extended'dan)."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'!!+', '!', text)
    text = re.sub(r'\?\?+', '?', text)
    text = re.sub(r'\.\.+', '.', text)
    contractions = {
        r"\byou're\b": 'you are', r"\bdon't\b": 'do not',
        r"\blet's\b": 'let us',  r"\bthat's\b": 'that is',
        r"\bit's\b":  'it is',
    }
    for pat, rep in contractions.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()


def _chunk_text_chatterbox(text: str, max_size: int = 300) -> list[str]:
    """Uzun metni Chatterbox için parçalara böl (MPT-Extended'dan)."""
    if len(text) <= max_size:
        return [text]
    chunks: list[str] = []
    sentences = re.split(r'([.!?])', text)
    current = ""
    for i in range(0, len(sentences), 2):
        sent = sentences[i].strip()
        punc = sentences[i + 1] if i + 1 < len(sentences) else ""
        full = sent + punc
        if len(current) + len(full) > max_size and current:
            chunks.append(current.strip())
            current = full
        else:
            current = (current + " " + full).strip() if current else full
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]


def _chatterbox_single(text: str, voice: str, output_path: Path) -> TTSResult:
    """Tek parça metin için Chatterbox TTS + WhisperX hizalaması (MPT-Extended'dan)."""
    global _chatterbox_model, _whisperx_model

    device = os.getenv("CHATTERBOX_DEVICE", "cpu").lower()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # voice: "chatterbox:clone:<name>" veya "chatterbox:default:..."
    parts = voice.split(":")
    voice_type      = parts[1] if len(parts) > 1 else "default"
    voice_base_name = parts[2].split("-")[0] if len(parts) > 2 else ""

    # Referans ses dosyası (klonlama için)
    audio_prompt = None
    if voice_type == "clone" and voice_base_name not in ("", "Voice Clone"):
        ref_dir = Path(__file__).parents[2] / "reference_audio"
        for ext in (".wav", ".mp3", ".flac", ".m4a"):
            candidate = ref_dir / (voice_base_name + ext)
            if candidate.exists():
                audio_prompt = str(candidate)
                break

    # Model yükleme
    if _chatterbox_model is None:
        try:
            _chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
        except Exception:
            device = "cpu"
            _chatterbox_model = ChatterboxTTS.from_pretrained(device=device)

    cfg_weight = float(os.getenv("CHATTERBOX_CFG_WEIGHT", "0.2"))
    wav = (_chatterbox_model.generate(text, audio_prompt_path=audio_prompt, cfg_weight=cfg_weight)
           if audio_prompt
           else _chatterbox_model.generate(text, cfg_weight=cfg_weight))

    # WAV → MP3
    tmp_wav = output_path.with_suffix(".chatterbox_tmp.wav")
    torchaudio.save(str(tmp_wav), wav, 24000)

    try:
        from moviepy import AudioFileClip as _AC
        _ac = _AC(str(tmp_wav))
        _ac.write_audiofile(str(output_path), logger=None)
        _ac.close()
        tmp_wav.unlink(missing_ok=True)
    except Exception:
        tmp_wav.rename(output_path)

    # WhisperX kelime hizalaması
    word_timestamps: list[dict] = []
    try:
        if _whisperx_model is None:
            compute_type = "int8" if device == "cpu" else "float16"
            _whisperx_model = whisperx.load_model("base", device, compute_type=compute_type)

        audio_arr = whisperx.load_audio(str(output_path))
        result    = _whisperx_model.transcribe(audio_arr, batch_size=16)

        if result and result.get("segments"):
            model_a, meta = whisperx.load_align_model(
                language_code=result["language"], device=device
            )
            result = whisperx.align(
                result["segments"], model_a, meta, audio_arr, device,
                return_char_alignments=False,
            )
            for seg in result.get("segments", []):
                for w in seg.get("words", []):
                    word  = w.get("word", "").strip()
                    start = w.get("start")
                    end   = w.get("end")
                    if word and start is not None and end is not None:
                        word_timestamps.append({
                            "word":  word,
                            "start": round(float(start), 3),
                            "end":   round(float(end), 3),
                        })
    except Exception as exc:
        log.warning("WhisperX hizalaması başarısız: %s", exc)

    duration = _probe_duration(output_path)
    log.info("Chatterbox TTS tamamlandı: %d kelime, %.1fs", len(word_timestamps), duration)
    return TTSResult(
        success=True,
        audio_path=str(output_path),
        duration_sec=duration,
        word_timestamps=word_timestamps,
    )


def _chatterbox_tts(text: str, voice: str, output_path: Path) -> TTSResult:
    """Chatterbox TTS — gerekirse metni parçalara böler (MPT-Extended'dan)."""
    if not CHATTERBOX_AVAILABLE:
        return TTSResult(success=False, error="chatterbox-tts veya whisperx kurulu değil")

    text = _preprocess_text_chatterbox(text.strip())
    threshold = int(os.getenv("CHATTERBOX_CHUNK_THRESHOLD", "600"))

    if len(text) <= threshold:
        return _chatterbox_single(text, voice, output_path)

    # Uzun metin → parçalara böl ve birleştir
    chunks = _chunk_text_for_chatterbox(text, max_size=300)
    tmp_paths: list[Path] = []
    all_words: list[dict] = []
    cum_dur = 0.0

    try:
        for i, chunk in enumerate(chunks):
            chunk_path = output_path.with_suffix(f".chunk{i}.mp3")
            chunk_result = _chatterbox_single(chunk, voice, chunk_path)
            if not chunk_result.success:
                return TTSResult(success=False, error=f"Parça {i} başarısız")
            tmp_paths.append(chunk_path)
            for w in chunk_result.word_timestamps:
                all_words.append({
                    "word":  w["word"],
                    "start": round(w["start"] + cum_dur, 3),
                    "end":   round(w["end"]   + cum_dur, 3),
                })
            cum_dur += chunk_result.duration_sec

        # MoviePy ile ses parçalarını birleştir
        from moviepy import AudioFileClip as _AC, concatenate_audioclips as _cat
        clips = [_AC(str(p)) for p in tmp_paths]
        combined = _cat(clips)
        combined.write_audiofile(str(output_path), logger=None)
        for c in clips:
            c.close()
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)

    duration = _probe_duration(output_path)
    return TTSResult(
        success=True,
        audio_path=str(output_path),
        duration_sec=duration,
        word_timestamps=all_words,
    )


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

    # ── MPT-Extended TTS seçenekleri ──
    if voice.startswith("chatterbox:"):
        result = _chatterbox_tts(text, voice, output_path)
        if not result.success:
            log.warning("Chatterbox başarısız, edge_tts'e geçiliyor: %s", result.error)
            result = _run_edge_tts(text, "en-US-GuyNeural", output_path)
        # Chatterbox zaten WhisperX ile hizalıyor; tekrar hizalamaya gerek yok
        return result

    if voice.startswith("siliconflow:"):
        result = _siliconflow_tts(text, voice, output_path)
        if not result.success:
            log.warning("SiliconFlow başarısız, edge_tts'e geçiliyor: %s", result.error)
            result = _run_edge_tts(text, "en-US-GuyNeural", output_path)

    elif voice.startswith("elevenlabs:"):
        result = _elevenlabs_tts(text, voice, output_path)
        if not result.success:
            log.warning("ElevenLabs başarısız, edge_tts'e geçiliyor: %s", result.error)
            result = _run_edge_tts(text, "en-GB-RyanNeural", output_path)

    elif language == "en":
        # Kokoro ses kodu mu (bm_, am_, bf_, af_ ile başlar)?
        _is_kokoro = any(voice.startswith(p) for p in ("bm_", "am_", "bf_", "af_"))
        if _is_kokoro:
            # Önce Kokoro dene (doğru ses kodu ile)
            result = _kokoro_tts(text, output_path, voice)
            if not result.success:
                log.warning("Kokoro başarısız, edge_tts varsayılan sesle devam: %s", result.error)
                result = _run_edge_tts(text, "en-GB-RyanNeural", output_path)
        else:
            # edge_tts ses kodu (en-GB-RyanNeural gibi)
            result = _run_edge_tts(text, voice, output_path)
            if not result.success:
                log.warning("edge_tts başarısız, Kokoro varsayılan sesle devam: %s", result.error)
                result = _kokoro_tts(text, output_path, "bm_george")
    else:
        # Türkçe: doğrudan gTTS (Kokoro İngilizce-only modeldir, Türkçe desteklemez)
        result = _gtts_tts(text, output_path, lang="tr")

    # faster-whisper ile kelime hizalaması (EN + TR)
    if result.success and align_words and not result.word_timestamps:
        result.word_timestamps = _align_with_whisper(Path(result.audio_path), text)

    return result
