"""
FFmpeg ile video işleme aracı: ken burns efekti, klip kesme, altyazı ekleme.
"""
import json
import logging
import subprocess
import textwrap
from pathlib import Path

from crewai.tools import BaseTool

log = logging.getLogger(__name__)


def _get_duration(path: Path) -> float:
    """ffprobe ile dosya süresini ölç."""
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


class FFmpegTool(BaseTool):
    name: str = "Video İşle"
    description: str = """
    Fotoğraf veya videoya sinematik efekt uygular.
    Input: JSON string {
        "input_path": "/tmp/...",
        "output_path": "/tmp/...",
        "type": "ken_burns|clip|subtitle",
        "duration": 7.0,
        "subtitle_text": "...",
        "zoom_direction": "in|out"
    }
    Output: JSON {"output_path": str, "duration": float, "success": bool}
    """

    def _run(self, input_str: str) -> str:
        """FFmpeg ile video işle."""
        try:
            data = json.loads(input_str)
            input_path = Path(data["input_path"])
            output_path = Path(data["output_path"])
            proc_type = data.get("type", "ken_burns")
            duration = float(data.get("duration", 7.0))
            subtitle_text = data.get("subtitle_text", "")
            zoom_direction = data.get("zoom_direction", "in")
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            log.error("FFmpegTool geçersiz input: %s", exc)
            return json.dumps({"success": False, "output_path": "", "duration": 0.0})

        if not input_path.exists():
            log.error("FFmpegTool: giriş dosyası bulunamadı: %s", input_path)
            return json.dumps({"success": False, "output_path": "", "duration": 0.0})

        output_path.parent.mkdir(parents=True, exist_ok=True)

        video_aspect = data.get("video_aspect", "16:9")

        try:
            if proc_type == "ken_burns":
                mood = data.get("mood", "neutral")
                self._ken_burns(input_path, output_path, duration, zoom_direction, video_aspect, mood)
            elif proc_type == "clip":
                self._clip(input_path, output_path, duration, data, video_aspect)
            elif proc_type == "subtitle":
                self._subtitle(input_path, output_path, subtitle_text)
            elif proc_type == "burn_srt":
                srt_path = Path(data.get("srt_path", ""))
                subtitle_color = data.get("subtitle_color", "#ffdc00")
                self._burn_srt(input_path, output_path, srt_path, subtitle_color)
            else:
                raise ValueError(f"Bilinmeyen işlem tipi: {proc_type}")

            actual_duration = _get_duration(output_path)
            log.info("FFmpeg işlemi tamamlandı: %s (%.1fs)", output_path.name, actual_duration)
            return json.dumps({
                "output_path": str(output_path),
                "duration": actual_duration,
                "success": True,
            })

        except subprocess.CalledProcessError as exc:
            log.error("FFmpegTool subprocess hatası: %s", exc.stderr.decode(errors="replace")[:500])
            return json.dumps({"success": False, "output_path": "", "duration": 0.0})
        except Exception as exc:
            log.error("FFmpegTool işlem hatası: %s", exc)
            return json.dumps({"success": False, "output_path": "", "duration": 0.0})

    @staticmethod
    def _aspect_to_res(aspect: str) -> tuple[int, int]:
        """Aspect ratio string → (width, height)."""
        if aspect == "9:16":
            return 1080, 1920
        elif aspect == "1:1":
            return 1080, 1080
        return 1920, 1080  # 16:9 default

    def _ken_burns(
        self,
        input_path: Path,
        output_path: Path,
        duration: float,
        zoom_direction: str,
        video_aspect: str = "16:9",
        mood: str = "neutral",
    ) -> None:
        """Fotoğrafa düzgün Ken Burns efekti uygula (titreme önlendi)."""
        fps = 25
        nb_frames = int(duration * fps)
        w, h = self._aspect_to_res(video_aspect)

        if zoom_direction == "in":
            zoom_expr = "zoom+0.0005"
        else:
            zoom_expr = "if(eq(on,1),1.3,zoom-0.0005)"

        # Mood-to-motion: sahne mood'una göre pan yönü
        _pan = {
            "dramatic":  ("iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),          # merkez zoom
            "tense":     ("iw*0.3-(iw/zoom/2)", "ih*0.3-(ih/zoom/2)"),       # sol üst → merkez
            "peaceful":  ("iw*0.6-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),         # yavaş sağdan sola
            "neutral":   ("iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),           # merkez
        }
        x_expr, y_expr = _pan.get(mood, _pan["neutral"])

        # Önce görseli büyüt, sonra zoompan uygula (titreme önlemi)
        vf = (
            f"scale=8000:-1,"
            f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}'"
            f":d={nb_frames}:s={w}x{h}:fps={fps},"
            f"scale={w}:{h}"
        )

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", str(input_path),
                "-vf", vf,
                "-t", str(duration),
                "-c:v", "libx264",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-an",
                str(output_path),
            ],
            check=True, capture_output=True,
            timeout=120,
        )

    def _clip(
        self,
        input_path: Path,
        output_path: Path,
        duration: float,
        data: dict,
        video_aspect: str = "16:9",
    ) -> None:
        """Video klibini kes ve yeniden boyutlandır."""
        start = float(data.get("clip_start", 0))
        w, h  = self._aspect_to_res(video_aspect)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", str(input_path),
                "-t", str(duration),
                "-vf", (
                    f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                    f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
                ),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-an",
                str(output_path),
            ],
            check=True, capture_output=True,
        )

    def _subtitle(
        self,
        input_path: Path,
        output_path: Path,
        text: str,
    ) -> None:
        """Video üzerine altyazı ekle."""
        wrapped = "\n".join(textwrap.wrap(text, width=60))
        escaped = wrapped.replace("'", "\\'").replace(":", "\\:").replace("\\n", "\n")

        drawtext_filter = (
            f"drawtext=text='{escaped}'"
            f":fontsize=40"
            f":fontcolor=white"
            f":x=(w-text_w)/2"
            f":y=h-th-40"
            f":box=1:boxcolor=black@0.5:boxborderw=10"
        )

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-vf", drawtext_filter,
                "-c:v", "libx264",
                "-c:a", "copy",
                str(output_path),
            ],
            check=True, capture_output=True,
        )


    @staticmethod
    def _hex_to_ass(hex_color: str) -> str:
        """#RRGGBB hex rengini ASS &HBBGGRR formatına çevir."""
        h = hex_color.lstrip("#")
        if len(h) != 6:
            return "&H00FFDC00"  # varsayılan sarı
        r, g, b = h[0:2], h[2:4], h[4:6]
        return f"&H00{b}{g}{r}".upper()

    def _burn_srt(
        self,
        input_path: Path,
        output_path: Path,
        srt_path: Path,
        subtitle_color: str = "#ffdc00",
    ) -> None:
        """SRT altyazısını video üzerine yak (libass gerektirir)."""
        if not srt_path.exists():
            raise FileNotFoundError(f"SRT dosyası bulunamadı: {srt_path}")

        ass_color = self._hex_to_ass(subtitle_color)
        # ffmpeg subtitles filtresi için path'i escape et
        escaped = str(srt_path).replace("\\", "/").replace(":", "\\\\:")
        style = (
            f"FontSize=38,Alignment=2,"
            f"PrimaryColour={ass_color},SecondaryColour=&H00FFFFFF,"
            f"BackColour=&H40000000,BorderStyle=1,"
            f"Outline=2,Shadow=1,MarginV=30,"
            f"Fontname=Arial,Bold=1"
        )

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-vf", f"subtitles='{escaped}':force_style='{style}'",
                "-c:v", "libx264",
                "-preset", "fast",
                "-c:a", "copy",
                str(output_path),
            ],
            check=True, capture_output=True,
            timeout=180,
        )


if __name__ == "__main__":
    print("✅ FFmpegTool modülü yüklendi")
    tool = FFmpegTool()
    print(f"   Tool adı: {tool.name}")
