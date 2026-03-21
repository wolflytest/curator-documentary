"""
Kokoro ONNX ile Türkçe TTS (text-to-speech) aracı.
Fallback: gTTS
"""
import json
import logging
import subprocess
import tempfile
from pathlib import Path

from crewai.tools import BaseTool

log = logging.getLogger(__name__)


def _get_duration(path: Path) -> float:
    """ffprobe ile ses dosyasının süresini ölç."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


class KokoroTTSTool(BaseTool):
    name: str = "Ses Üret"
    description: str = """
    Türkçe metin için profesyonel seslendirme üretir.
    Input: JSON string {
        "text": "Seslendirme metni",
        "output_path": "/tmp/...",
        "voice": "af_heart",
        "speed": 0.95
    }
    Output: JSON {"output_path": str, "duration_sec": float, "success": bool}
    """

    def _run(self, input_str: str) -> str:
        """Metinden ses dosyası üret."""
        try:
            data = json.loads(input_str)
            text = data.get("text", "")
            output_path = Path(data["output_path"])
            voice = data.get("voice", "af_heart")
            speed = float(data.get("speed", 0.95))
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            log.error("KokoroTTSTool geçersiz input: %s", exc)
            return json.dumps({"success": False, "output_path": "", "duration_sec": 0.0})

        if not text.strip():
            log.warning("KokoroTTSTool: boş metin.")
            return json.dumps({"success": False, "output_path": "", "duration_sec": 0.0})

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Önce Kokoro dene
        try:
            if self._kokoro_tts(text, output_path, voice, speed):
                duration = _get_duration(output_path)
                log.info("Kokoro TTS başarılı: %.1fs", duration)
                return json.dumps({
                    "output_path": str(output_path),
                    "duration_sec": duration,
                    "success": True,
                })
        except Exception as exc:
            log.warning("Kokoro TTS başarısız (%s), gTTS fallback deneniyor...", exc)

        # Fallback: gTTS
        try:
            self._gtts_fallback(text, output_path)
            duration = _get_duration(output_path)
            log.info("gTTS fallback başarılı: %.1fs", duration)
            return json.dumps({
                "output_path": str(output_path),
                "duration_sec": duration,
                "success": True,
            })
        except Exception as exc:
            log.error("KokoroTTSTool gTTS de başarısız: %s", exc)
            return json.dumps({"success": False, "output_path": "", "duration_sec": 0.0})

    def _kokoro_tts(
        self,
        text: str,
        output_path: Path,
        voice: str,
        speed: float,
    ) -> bool:
        """Kokoro ONNX ile ses üret, mp3 olarak kaydet."""
        import soundfile as sf
        from kokoro_onnx import Kokoro

        # Model dosyaları mevcut mu kontrol et
        model_path = Path("kokoro-v1.9.onnx")
        voices_path = Path("voices-v1.0.bin")
        if not model_path.exists() or not voices_path.exists():
            raise FileNotFoundError("Kokoro model dosyaları bulunamadı (kokoro-v1.9.onnx, voices-v1.0.bin)")

        kokoro = Kokoro(str(model_path), str(voices_path))
        samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="tr")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        try:
            sf.write(str(wav_path), samples, sample_rate)
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(wav_path),
                    "-ar", "16000",
                    "-ac", "1",
                    "-b:a", "128k",
                    str(output_path),
                ],
                check=True, capture_output=True,
            )
        finally:
            wav_path.unlink(missing_ok=True)

        return True

    def _gtts_fallback(self, text: str, output_path: Path) -> None:
        """gTTS ile Türkçe ses üret, mp3 olarak kaydet."""
        from gtts import gTTS
        tts = gTTS(text=text, lang="tr")
        tts.save(str(output_path))


if __name__ == "__main__":
    print("✅ KokoroTTSTool modülü yüklendi")
    tool = KokoroTTSTool()
    print(f"   Tool adı: {tool.name}")
