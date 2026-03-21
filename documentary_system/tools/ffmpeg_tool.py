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

        try:
            if proc_type == "ken_burns":
                self._ken_burns(input_path, output_path, duration, zoom_direction)
            elif proc_type == "clip":
                self._clip(input_path, output_path, duration, data)
            elif proc_type == "subtitle":
                self._subtitle(input_path, output_path, subtitle_text)
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

    def _ken_burns(
        self,
        input_path: Path,
        output_path: Path,
        duration: float,
        zoom_direction: str,
    ) -> None:
        """Fotoğrafa Ken Burns (zoom+pan) efekti uygula."""
        fps = 25
        nb_frames = int(duration * fps)

        if zoom_direction == "in":
            zoom_expr = "zoom+0.0015"
        else:
            zoom_expr = "if(eq(on,1),1.5,zoom-0.0015)"

        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"

        vf = (
            f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}'"
            f":d={nb_frames}:s=1920x1080:fps={fps},"
            f"scale=1920:1080"
        )

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", str(input_path),
                "-vf", vf,
                "-t", str(duration),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-an",
                str(output_path),
            ],
            check=True, capture_output=True,
        )

    def _clip(
        self,
        input_path: Path,
        output_path: Path,
        duration: float,
        data: dict,
    ) -> None:
        """Video klibini kes ve yeniden boyutlandır."""
        start = float(data.get("clip_start", 0))
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", str(input_path),
                "-t", str(duration),
                "-vf", (
                    "scale=1920:1080:force_original_aspect_ratio=decrease,"
                    "pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
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


if __name__ == "__main__":
    print("✅ FFmpegTool modülü yüklendi")
    tool = FFmpegTool()
    print(f"   Tool adı: {tool.name}")
