"""
Altyazı servisi — MoneyPrinterTurbo-Extended kodu + karaoke overlay.

MPT-Extended'dan alınan fonksiyonlar (orijinal kod):
  - levenshtein_distance / similarity / correct  → SRT drift düzeltme
  - _wrap_text_into_lines / _balance_subtitle_lines → daha iyi satır bölme

Curator eklentileri:
  - build_srt()           → word_timestamps → SRT
  - render_karaoke_overlay() → PIL karaoke animasyonu
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

# Karaoke stil sabitleri
_FONT_SIZE_ACTIVE   = 52
_FONT_SIZE_NORMAL   = 40
_COLOR_ACTIVE       = (255, 220, 0)    # Sarı
_COLOR_NORMAL       = (255, 255, 255)  # Beyaz
_SHADOW_COLOR       = (0, 0, 0)
_BG_ALPHA           = 160             # 0–255
_MARGIN_BOTTOM      = 60
_MAX_LINE_CHARS     = 50


# ── SRT üretimi ───────────────────────────────────────────────────────────────

def build_srt(word_timestamps: list[dict], output_path: Path) -> Path:
    """
    Kelime zaman damgalarından SRT dosyası üret.
    Maksimum 55 karakter/satır, eşit satır dağılımı.
    """
    from documentary_system.tools.srt_generator import _split_into_chunks, _ts

    if not word_timestamps:
        output_path.write_text("", encoding="utf-8")
        return output_path

    # Kelimeleri grupla
    all_words  = [w["word"] for w in word_timestamps]
    chunks     = _split_into_chunks(all_words, max_chars=55)
    total_dur  = word_timestamps[-1]["end"] if word_timestamps else 1.0

    # Her chunk için zaman damgası: ilk/son kelime indexine bak
    word_idx   = 0
    lines: list[str] = []

    for i, chunk in enumerate(chunks):
        chunk_words = chunk.split()
        n = len(chunk_words)

        # Başlangıç: ilk kelimenin start, bitiş: son kelimenin end
        start_ts = word_timestamps[word_idx]["start"] if word_idx < len(word_timestamps) else 0.0
        end_idx  = min(word_idx + n - 1, len(word_timestamps) - 1)
        end_ts   = word_timestamps[end_idx]["end"]

        lines += [
            str(i + 1),
            f"{_ts(start_ts)} --> {_ts(end_ts)}",
            chunk,
            "",
        ]
        word_idx += n

    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("SRT üretildi: %s (%d satır)", output_path.name, len(chunks))
    return output_path


# ── Karaoke render ─────────────────────────────────────────────────────────────

def render_karaoke_overlay(
    word_timestamps: list[dict],
    video_duration: float,
    output_path: Path,
    width: int = 1920,
    height: int = 1080,
    fps: int = 25,
) -> Path | None:
    """
    Karaoke-style altyazı overlay video oluştur (şeffaf arka plan yoksa siyah maske).
    Aktif kelime sarı, diğerleri beyaz.

    Returns:
        output_path (mp4) veya None (hata)
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        try:
            from moviepy import ImageSequenceClip
        except ImportError:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        if not word_timestamps:
            return None

        font_normal = _load_font(_FONT_SIZE_NORMAL)
        font_active = _load_font(_FONT_SIZE_ACTIVE)

        total_frames = int(video_duration * fps)
        frames: list[np.ndarray] = []

        # Aktif kelime cache: frame_t → word_index
        for frame_i in range(total_frames):
            t = frame_i / fps
            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            # Hangi satır grubunu göster?
            active_idx = _word_index_at(word_timestamps, t)
            if active_idx < 0:
                frames.append(np.array(img.convert("RGB")))
                continue

            group = _get_display_group(word_timestamps, active_idx, max_chars=_MAX_LINE_CHARS)
            _draw_karaoke_line(draw, group, active_idx, font_normal, font_active, width, height)

            frames.append(np.array(img.convert("RGB")))

        if not frames:
            return None

        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(str(output_path), fps=fps, codec="libx264",
                             ffmpeg_params=["-pix_fmt", "yuva420p", "-an"],
                             logger=None)
        log.info("Karaoke overlay üretildi: %s", output_path.name)
        return output_path

    except Exception as exc:
        log.warning("Karaoke render hatası: %s", exc)
        return None


# ── Yardımcı ─────────────────────────────────────────────────────────────────

def _load_font(size: int):
    """PIL font yükle: DejaVu Sans → fallback default."""
    from PIL import ImageFont
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for path in font_candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _word_index_at(words: list[dict], t: float) -> int:
    """t anındaki aktif kelime indexini döndür. Yok → -1."""
    for i, w in enumerate(words):
        if w["start"] <= t < w["end"]:
            return i
    # Henüz başlamadı mı?
    if words and t < words[0]["start"]:
        return -1
    # Bitti mi?
    return -1


def _get_display_group(words: list[dict], active_idx: int, max_chars: int) -> list[dict]:
    """
    Aktif kelimenin çevresindeki görüntülenecek kelime grubunu belirle.
    Maksimum max_chars karakter sığacak kadar kelime al.
    """
    # Aktif kelimeden geriye ve ileriye doğru grup oluştur
    group = []
    char_count = 0

    # Aktif kelimenin satır başlangıcını bul
    start = active_idx
    while start > 0:
        candidate = words[start - 1]["word"]
        if char_count + len(candidate) + 1 > max_chars // 2:
            break
        start -= 1
        char_count += len(candidate) + 1

    # start'tan itibaren max_chars'a kadar ekle
    char_count = 0
    for i in range(start, len(words)):
        w = words[i]
        add = len(w["word"]) + (1 if group else 0)
        if char_count + add > max_chars and group:
            break
        group.append({**w, "_idx": i})
        char_count += add

    return group


def _draw_karaoke_line(
    draw,
    group: list[dict],
    active_idx: int,
    font_normal,
    font_active,
    width: int,
    height: int,
) -> None:
    """Tek satırı karaoke renkleriyle çiz."""
    from PIL import ImageDraw

    # Önce toplam genişliği hesapla
    total_w = 0
    parts = []
    for item in group:
        is_active = item["_idx"] == active_idx
        font  = font_active if is_active else font_normal
        color = _COLOR_ACTIVE if is_active else _COLOR_NORMAL
        bbox  = font.getbbox(item["word"] + " ")
        w, h  = bbox[2] - bbox[0], bbox[3] - bbox[1]
        parts.append({"text": item["word"], "font": font, "color": color, "w": w, "h": h})
        total_w += w

    if not parts:
        return

    # Siyah yarı şeffaf arka plan kutusu
    max_h   = max(p["h"] for p in parts) + 20
    box_x0  = max(0, width // 2 - total_w // 2 - 20)
    box_y0  = height - _MARGIN_BOTTOM - max_h
    box_x1  = min(width, width // 2 + total_w // 2 + 20)
    box_y1  = height - _MARGIN_BOTTOM
    draw.rectangle([box_x0, box_y0, box_x1, box_y1],
                   fill=(0, 0, 0, _BG_ALPHA))

    # Kelimeleri sırayla çiz
    x = width // 2 - total_w // 2
    y = box_y0 + 10
    for p in parts:
        # Gölge
        draw.text((x + 2, y + 2), p["text"], font=p["font"], fill=_SHADOW_COLOR + (180,))
        # Asıl metin
        draw.text((x, y), p["text"], font=p["font"], fill=p["color"] + (255,))
        x += p["w"]

# ── MPT-Extended: Levenshtein SRT düzeltme (orijinal kod) ────────────────────
# Kaynak: https://github.com/Asad-Ismail/MoneyPrinterTurbo-Extended/blob/main/app/services/subtitle.py

def _file_to_subtitles(filename: str | Path) -> list[tuple]:
    """SRT dosyasını parse et."""
    path = Path(filename)
    if not path.exists():
        return []
    times_texts = []
    current_times = None
    current_text = ""
    index = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            times = re.findall(r"([0-9]*:[0-9]*:[0-9]*,[0-9]*)", line)
            if times:
                current_times = line
            elif line.strip() == "" and current_times:
                index += 1
                times_texts.append((index, current_times.strip(), current_text.strip()))
                current_times, current_text = None, ""
            elif current_times:
                current_text += line
    return times_texts


def levenshtein_distance(s1: str, s2: str) -> int:
    """Levenshtein mesafesi (MPT-Extended orijinal)."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _subtitle_similarity(a: str, b: str) -> float:
    """Levenshtein tabanlı metin benzerliği (0–1)."""
    distance = levenshtein_distance(a.lower(), b.lower())
    max_length = max(len(a), len(b))
    return 1 - (distance / max_length) if max_length else 1.0


def correct_srt(subtitle_file: str | Path, video_script: str) -> None:
    """
    SRT'yi video senaryosu ile karşılaştırarak Levenshtein tabanlı düzeltme.
    MPT-Extended orijinal `correct()` fonksiyonu — sadece import değişti.
    """
    subtitle_items = _file_to_subtitles(subtitle_file)
    # Noktalama ile böl
    script_lines = [s.strip() for s in re.split(r'[.!?。！？]', video_script) if s.strip()]

    corrected = False
    new_subtitle_items = []
    script_index = 0
    subtitle_index = 0

    while script_index < len(script_lines) and subtitle_index < len(subtitle_items):
        script_line = script_lines[script_index]
        subtitle_line = subtitle_items[subtitle_index][2].strip()

        if script_line == subtitle_line:
            new_subtitle_items.append(subtitle_items[subtitle_index])
            script_index += 1
            subtitle_index += 1
        else:
            combined = subtitle_line
            start_time = subtitle_items[subtitle_index][1].split(" --> ")[0]
            end_time   = subtitle_items[subtitle_index][1].split(" --> ")[1]
            next_idx   = subtitle_index + 1

            while next_idx < len(subtitle_items):
                next_sub = subtitle_items[next_idx][2].strip()
                if _subtitle_similarity(script_line, combined + " " + next_sub) > _subtitle_similarity(script_line, combined):
                    combined += " " + next_sub
                    end_time  = subtitle_items[next_idx][1].split(" --> ")[1]
                    next_idx += 1
                else:
                    break

            new_subtitle_items.append((
                len(new_subtitle_items) + 1,
                f"{start_time} --> {end_time}",
                script_line if _subtitle_similarity(script_line, combined) > 0.8 else combined,
            ))
            corrected = True
            script_index  += 1
            subtitle_index = next_idx

    if corrected:
        with open(subtitle_file, "w", encoding="utf-8") as fd:
            for i, item in enumerate(new_subtitle_items):
                fd.write(f"{i + 1}\n{item[1]}\n{item[2]}\n\n")
        log.info("SRT Levenshtein düzeltmesi tamamlandı: %s", Path(subtitle_file).name)


# ── MPT-Extended: Satır dengeleme (orijinal kod) ─────────────────────────────

def _wrap_text_into_lines(text: str, max_chars_per_line: int = 40, max_lines: int = 2) -> list[str]:
    """Metni max_chars_per_line sınırına göre satırlara böl. Virgül noktası dikkate alınır."""
    comma_segments = [s.strip() for s in text.split(',') if s.strip()]
    lines = []
    current_line = ""

    for segment in comma_segments:
        for word in segment.split():
            test_line = current_line + (" " if current_line else "") + word
            if len(test_line) <= max_chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    lines.append(word)
                    current_line = ""
                if len(lines) >= max_lines:
                    break
        if current_line and len(current_line) > max_chars_per_line * 0.6:
            lines.append(current_line)
            current_line = ""
            if len(lines) >= max_lines:
                break

    if current_line and len(lines) < max_lines:
        lines.append(current_line)

    if len(lines) > 1:
        lines = _balance_subtitle_lines(lines, max_chars_per_line)
    return lines


def _balance_subtitle_lines(lines: list[str], max_chars_per_line: int) -> list[str]:
    """Satır uzunluklarını ortalama hizalama için dengele (MPT-Extended orijinal)."""
    if len(lines) <= 1:
        return lines
    balanced = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            words_current = line.split()
            words_next    = lines[i + 1].split()
            if len(line) < max_chars_per_line * 0.7 and len(words_next) > 1:
                test = line + " " + words_next[0]
                if len(test) <= max_chars_per_line:
                    balanced.append(test)
                    lines[i + 1] = " ".join(words_next[1:])
                    continue
        balanced.append(line)
    return balanced
