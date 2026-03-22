"""
SRT altyazı dosyası üretici.

Narration metni + TTS süresi → SRT dosyası.
Kelime bazlı eşit zaman dağılımı yapar.
"""
from pathlib import Path


def generate_srt(text: str, duration_sec: float, output_path: Path) -> Path:
    """
    Metni SRT formatına çevir.

    - Her chunk maksimum 55 karakter (2 satır 27-28 karakter)
    - Süre boyunca eşit dağılım (kelime bazlı)
    - Çok kısa metinler için minimum 1.0s/chunk

    Returns:
        output_path (yazılan dosya)
    """
    words = text.strip().split()
    if not words or duration_sec <= 0:
        output_path.write_text("", encoding="utf-8")
        return output_path

    chunks = _split_into_chunks(words, max_chars=55)
    if not chunks:
        output_path.write_text("", encoding="utf-8")
        return output_path

    time_per_chunk = max(duration_sec / len(chunks), 1.0)

    lines: list[str] = []
    for i, chunk in enumerate(chunks):
        start = i * time_per_chunk
        end   = min((i + 1) * time_per_chunk, duration_sec)
        lines += [
            str(i + 1),
            f"{_ts(start)} --> {_ts(end)}",
            chunk,
            "",
        ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _split_into_chunks(words: list[str], max_chars: int = 55) -> list[str]:
    """Kelime listesini max_chars sınırını aşmayan chunklara böl."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        # +1 boşluk için (ilk kelime hariç)
        add_len = len(word) + (1 if current else 0)
        if current_len + add_len > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += add_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def _ts(seconds: float) -> str:
    """Saniyeyi SRT timestamp formatına çevir: HH:MM:SS,mmm"""
    h   = int(seconds // 3600)
    m   = int((seconds % 3600) // 60)
    s   = int(seconds % 60)
    ms  = int(round((seconds % 1) * 1000))
    if ms == 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


if __name__ == "__main__":
    from pathlib import Path
    out = Path("/tmp/test_sub.srt")
    generate_srt(
        "The Ottoman Empire rose to power in the 14th century, "
        "conquering vast territories across three continents. "
        "At its peak it was one of the most powerful states in the world.",
        duration_sec=15.0,
        output_path=out,
    )
    print(out.read_text())
