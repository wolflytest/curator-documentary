"""
Ana orkestratör: tüm crew'ları koordine eder.
DocumentaryState üzerinden çalışır, her adımı SQLite'a kaydeder.

Medya seçimi:
  - Eski: CrewAI media_crew (ajan tabanlı, tutarsız)
  - Yeni: media_selector (programatik, semantik Gemini Vision skoru)
"""
import json
import logging
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

import db
from documentary_system.crews.qa_crew import create_qa_crew
from documentary_system.crews.script_crew import create_script_crew
from documentary_system.state.documentary_state import DocumentaryState, SceneState
from documentary_system.tools.ffmpeg_tool import FFmpegTool
from documentary_system.tools.kokoro_tts_tool import KokoroTTSTool
from documentary_system.tools.media_selector import select_best_media
from documentary_system.tools.srt_generator import generate_srt

log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "documentary_output"
OUTPUT_DIR.mkdir(exist_ok=True)

BG_MUSIC_DIR = Path(__file__).parent.parent / "bg_music"

# Sahne arası bekleme: programatik selector ile çok daha az Gemini çağrısı olduğundan kısaltıldı
_SCENE_RATE_LIMIT_SECS = 20


def _update_status(state: DocumentaryState, status: str) -> None:
    state.status = status
    db.update_documentary_status(state.doc_id, status)
    log.info("[Belgesel #%d] Durum: %s", state.doc_id, status)


def _apply_script(state: DocumentaryState, script_data: dict) -> DocumentaryState:
    state.title       = script_data.get("title") or state.topic
    state.description = script_data.get("description", "")
    state.tags        = script_data.get("tags", [])
    state.script_revision_count = script_data.get("revision_count", 0)
    state.script_critic_notes   = script_data.get("critic_notes", "")

    scenes = []
    for i, s in enumerate(script_data.get("scenes", [])):
        scenes.append(SceneState(
            index=s.get("index", i),
            narration=s.get("narration", ""),
            search_keywords=s.get("search_keywords", []),
            visual_description=s.get("visual_description", ""),
            mood=s.get("mood", "neutral"),
            transition=s.get("transition", "fade"),
            duration_sec=float(s.get("duration_sec", 7.0)),
        ))
    state.scenes = scenes
    log.info("[Belgesel #%d] Senaryo uygulandı: %d sahne", state.doc_id, len(scenes))
    return state


def _get_duration(path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _combine_clip_audio(clip_path: Path, audio_path: Path, output_path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y",
         "-i", str(clip_path),
         "-i", str(audio_path),
         "-map", "0:v:0",
         "-map", "1:a:0",
         "-c:v", "copy",
         "-c:a", "aac",
         "-b:a", "128k",
         "-shortest",
         str(output_path)],
        check=True, capture_output=True,
    )
    return output_path


def _find_background_music() -> Path | None:
    """bg_music/ klasöründe rastgele bir müzik dosyası seç."""
    if not BG_MUSIC_DIR.exists():
        return None
    music_files = list(BG_MUSIC_DIR.glob("*.mp3")) + list(BG_MUSIC_DIR.glob("*.wav"))
    if not music_files:
        return None
    chosen = random.choice(music_files)
    log.info("Arka plan müziği: %s", chosen.name)
    return chosen


def _add_background_music(video_path: Path, music_path: Path, work_dir: Path) -> Path | None:
    """Videoyu arka plan müziğiyle karıştır (-25dB müzik, sonsuz döngü)."""
    output_path = work_dir / f"withmusic_{video_path.name}"
    try:
        subprocess.run(
            ["ffmpeg", "-y",
             "-i", str(video_path),
             "-stream_loop", "-1",
             "-i", str(music_path),
             "-filter_complex",
             "[1:a]volume=0.12,apad[music];[0:a][music]amix=inputs=2:duration=first[aout]",
             "-map", "0:v",
             "-map", "[aout]",
             "-c:v", "copy",
             "-c:a", "aac",
             "-b:a", "192k",
             "-shortest",
             str(output_path)],
            check=True, capture_output=True,
        )
        log.info("Arka plan müziği eklendi: %s", output_path.name)
        return output_path
    except subprocess.CalledProcessError as exc:
        log.warning("Arka plan müziği eklenemedi: %s", exc.stderr.decode(errors="replace")[:200])
        return None


def _concat_scenes(state: DocumentaryState, work_dir: Path) -> Path:
    combined_clips = []
    for scene in state.scenes:
        if not scene.final_clip_path or not scene.tts_path:
            log.warning("Sahne %d eksik (clip=%s, tts=%s), atlanıyor.",
                        scene.index, scene.final_clip_path, scene.tts_path)
            continue

        clip_path    = Path(scene.final_clip_path)
        tts_path     = Path(scene.tts_path)
        combined_path = work_dir / f"combined_{scene.index:03d}.mp4"

        try:
            _combine_clip_audio(clip_path, tts_path, combined_path)
            combined_clips.append(combined_path)
            log.info("Sahne %d birleştirildi: %s", scene.index, combined_path.name)
        except subprocess.CalledProcessError as exc:
            log.error("Sahne %d birleştirme hatası: %s",
                      scene.index, exc.stderr.decode(errors="replace")[:300])

    if not combined_clips:
        raise RuntimeError("Birleştirilecek tamamlanmış sahne bulunamadı.")

    concat_file = work_dir / "concat_list.txt"
    concat_file.write_text(
        "\n".join(f"file '{p}'" for p in combined_clips),
        encoding="utf-8",
    )

    safe_title  = re.sub(r"[^\w\-_\u0080-\uffff]", "_", state.title or state.topic)[:50]
    output_path = OUTPUT_DIR / f"{state.doc_id}_{safe_title}.mp4"

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

    duration = _get_duration(output_path)
    log.info("Final video oluşturuldu: %s (%.1fs)", output_path.name, duration)
    return output_path


def _extract_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    log.error("JSON çıkarılamadı. Ham çıktı (ilk 500): %s", raw[:500])
    raise ValueError("LLM çıktısından JSON çıkarılamadı.")


def run_documentary(
    topic: str,
    target_duration: int = 600,
    language: str = "tr",
    voice: str = "gtts_tr",
) -> dict:
    """
    Tam belgesel üretim pipeline'ı.

    Adımlar:
      1. script_crew  → senaryo (target language'da)
      2. media_selector × sahne  → semantik görsel eşleştirme
      3. KokoroTTS × sahne  → ses
      4. SRT üret × sahne  → altyazı
      5. FFmpeg × sahne  → klip + altyazı yak
      6. Sahneleri birleştir
      7. Arka plan müziği (bg_music/ varsa)
      8. QA crew

    Args:
        topic:           Belgesel konusu
        target_duration: Hedef süre (saniye)
        language:        'tr' | 'en'
        voice:           Kokoro ses kodu (en için) veya 'gtts_tr' (tr için)
    """
    work_dir = Path(f"/tmp/curator_docs/{int(time.time())}")
    work_dir.mkdir(parents=True, exist_ok=True)
    log.info("Belgesel başlatılıyor: '%s' lang=%s voice=%s", topic, language, voice)

    doc_id = db.create_documentary(topic)
    state  = DocumentaryState(doc_id=doc_id, topic=topic)

    try:
        # ── 1. Senaryo ────────────────────────────────────────────────────────
        _update_status(state, "scripting")
        script_crew, _ = create_script_crew(topic, target_duration, language)
        script_result  = script_crew.kickoff()
        script_data    = _extract_json(script_result.raw)
        state          = _apply_script(state, script_data)
        db.update_documentary_status(doc_id, "scripting", script_json=state.to_json())
        log.info("[#%d] Senaryo: '%s', %d sahne", doc_id, state.title, len(state.scenes))

        # ── 2. Semantik medya seçimi ──────────────────────────────────────────
        _update_status(state, "searching")
        for scene in state.scenes:
            if scene.index > 0:
                log.info("[#%d] Rate limit önlemi: %ds bekleniyor...", doc_id, _SCENE_RATE_LIMIT_SECS)
                time.sleep(_SCENE_RATE_LIMIT_SECS)
            log.info("[#%d] Sahne %d medya seçiliyor...", doc_id, scene.index)
            try:
                selected = select_best_media(scene, state)
                if selected:
                    scene.approved_media = selected
                    scene.color_palette  = selected.get("dominant_colors", [])
                    source_url = selected.get("source_url", "")
                    if source_url:
                        state.used_media_hashes.add(source_url[:64])
                    db.save_documentary_media(
                        documentary_id=doc_id,
                        scene_index=scene.index,
                        source=selected.get("source", ""),
                        source_url=selected.get("source_url", ""),
                        local_path=selected.get("local_path", ""),
                        relevance_score=selected.get("relevance_score", 0.0),
                    )
                    log.info("[#%d] Sahne %d medya: rel=%.1f qual=%.1f",
                             doc_id, scene.index,
                             selected.get("relevance_score", 0),
                             selected.get("quality_score", 0))
                else:
                    log.warning("[#%d] Sahne %d için medya bulunamadı.", doc_id, scene.index)
            except Exception as exc:
                log.error("[#%d] Sahne %d medya hatası: %s", doc_id, scene.index, exc)

        # ── 3. TTS + SRT + FFmpeg ─────────────────────────────────────────────
        _update_status(state, "assembling")
        tts_tool    = KokoroTTSTool()
        ffmpeg_tool = FFmpegTool()

        for scene in state.scenes:
            # TTS
            tts_output = work_dir / f"tts_{scene.index:03d}.mp3"
            tts_result = json.loads(tts_tool._run(json.dumps({
                "text":        scene.narration,
                "output_path": str(tts_output),
                "voice":       voice if language == "en" else "gtts_tr",
                "language":    language,
                "speed":       0.95,
            })))
            tts_duration = 0.0
            if tts_result.get("success"):
                scene.tts_path = tts_result["output_path"]
                tts_duration   = tts_result.get("duration_sec", 0.0)
                log.info("[#%d] Sahne %d TTS: %.1fs", doc_id, scene.index, tts_duration)
            else:
                log.warning("[#%d] Sahne %d TTS başarısız.", doc_id, scene.index)

            # SRT altyazı üret
            srt_path = work_dir / f"sub_{scene.index:03d}.srt"
            audio_dur = tts_duration if tts_duration > 0 else scene.duration_sec
            generate_srt(scene.narration, audio_dur, srt_path)

            # FFmpeg klip
            clip_output    = work_dir / f"clip_{scene.index:03d}.mp4"
            clip_with_sub  = work_dir / f"clip_sub_{scene.index:03d}.mp4"

            if scene.approved_media and scene.approved_media.get("local_path"):
                media       = scene.approved_media
                ffmpeg_type = "ken_burns" if media.get("media_type") == "photo" else "clip"

                clip_result = json.loads(ffmpeg_tool._run(json.dumps({
                    "input_path":     media.get("local_path", ""),
                    "output_path":    str(clip_output),
                    "type":           ffmpeg_type,
                    "duration":       scene.duration_sec,
                    "clip_start":     media.get("clip_start", 0.0),
                    "zoom_direction": "in" if scene.index % 2 == 0 else "out",
                })))

                if clip_result.get("success"):
                    # Altyazı yak
                    if srt_path.exists() and srt_path.stat().st_size > 10:
                        sub_result = json.loads(ffmpeg_tool._run(json.dumps({
                            "input_path":  clip_result["output_path"],
                            "output_path": str(clip_with_sub),
                            "type":        "burn_srt",
                            "srt_path":    str(srt_path),
                        })))
                        scene.final_clip_path = (
                            sub_result["output_path"]
                            if sub_result.get("success")
                            else clip_result["output_path"]
                        )
                    else:
                        scene.final_clip_path = clip_result["output_path"]
                    log.info("[#%d] Sahne %d klip + altyazı hazır.", doc_id, scene.index)
                else:
                    log.warning("[#%d] Sahne %d FFmpeg başarısız.", doc_id, scene.index)
            else:
                # Medya yok → siyah arka plan
                log.warning("[#%d] Sahne %d için medya yok, siyah arka plan...", doc_id, scene.index)
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-f", "lavfi",
                         "-i", f"color=black:size=1920x1080:duration={scene.duration_sec}:rate=25",
                         "-c:v", "libx264", "-pix_fmt", "yuv420p",
                         str(clip_output)],
                        check=True, capture_output=True,
                    )
                    # Altyazı yak (siyah zemin üstüne)
                    if srt_path.exists() and srt_path.stat().st_size > 10:
                        sub_result = json.loads(ffmpeg_tool._run(json.dumps({
                            "input_path":  str(clip_output),
                            "output_path": str(clip_with_sub),
                            "type":        "burn_srt",
                            "srt_path":    str(srt_path),
                        })))
                        scene.final_clip_path = (
                            sub_result["output_path"]
                            if sub_result.get("success")
                            else str(clip_output)
                        )
                    else:
                        scene.final_clip_path = str(clip_output)
                except subprocess.CalledProcessError as exc:
                    log.error("[#%d] Sahne %d siyah klip hatası: %s",
                              doc_id, scene.index, exc.stderr.decode(errors="replace")[:300])

        # ── 4. Sahneleri birleştir ────────────────────────────────────────────
        output_path  = _concat_scenes(state, work_dir)
        state.output_path = str(output_path)

        # ── 5. Arka plan müziği (opsiyonel) ───────────────────────────────────
        bg_music = _find_background_music()
        if bg_music:
            with_music = _add_background_music(output_path, bg_music, OUTPUT_DIR)
            if with_music:
                output_path.unlink(missing_ok=True)
                output_path = with_music
                state.output_path = str(output_path)

        # ── 6. QA ─────────────────────────────────────────────────────────────
        _update_status(state, "qa")
        qa_crew  = create_qa_crew(state)
        qa_result = qa_crew.kickoff()
        try:
            qa_data = _extract_json(qa_result.raw)
        except Exception:
            qa_data = {"overall_score": 5.0, "approved": True, "technical_issues": []}

        qa_score = float(qa_data.get("overall_score", 5.0))
        log.info("[#%d] QA: skor=%.1f", doc_id, qa_score)

        # ── 7. Tamamlandı ─────────────────────────────────────────────────────
        _update_status(state, "done")
        db.update_documentary_status(doc_id, "done", output_path=str(output_path))

        return {
            "doc_id":       doc_id,
            "title":        state.title,
            "output_path":  str(output_path),
            "scene_count":  len(state.scenes),
            "qa_score":     qa_score,
            "status":       "done",
        }

    except Exception as exc:
        log.error("[#%d] Belgesel üretim hatası: %s", doc_id, exc, exc_info=True)
        db.update_documentary_status(doc_id, "error", error_msg=str(exc))
        raise

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        log.info("[#%d] Geçici dosyalar temizlendi.", doc_id)


if __name__ == "__main__":
    import db as _db
    _db.init_db()
    print("✅ Orchestrator hazır.")
    print("   result = run_documentary('Osmanlı İmparatorluğu', target_duration=120)")
