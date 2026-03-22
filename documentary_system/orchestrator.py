"""
Ana orkestratör: tüm crew'ları koordine eder.
DocumentaryState üzerinden çalışır, her adımı SQLite'a kaydeder.
"""
import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path

import db
from documentary_system.crews.media_crew import create_media_crew
from documentary_system.crews.qa_crew import create_qa_crew
from documentary_system.crews.script_crew import create_script_crew
from documentary_system.state.documentary_state import DocumentaryState, SceneState
from documentary_system.tools.ffmpeg_tool import FFmpegTool
from documentary_system.tools.kokoro_tts_tool import KokoroTTSTool

log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "documentary_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _update_status(state: DocumentaryState, status: str) -> None:
    """State'i ve DB'yi güncelle."""
    state.status = status
    db.update_documentary_status(state.doc_id, status)
    log.info("[Belgesel #%d] Durum: %s", state.doc_id, status)


def _apply_script(state: DocumentaryState, script_data: dict) -> DocumentaryState:
    """Script crew çıktısını state'e uygula."""
    state.title = script_data.get("title") or state.topic
    state.description = script_data.get("description", "")
    state.tags = script_data.get("tags", [])
    state.script_revision_count = script_data.get("revision_count", 0)
    state.script_critic_notes = script_data.get("critic_notes", "")

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


def _apply_media(
    state: DocumentaryState,
    scene: SceneState,
    media_data: dict,
) -> None:
    """Medya crew çıktısını sahneye uygula."""
    selected = media_data.get("selected_media")
    if not selected:
        log.warning("[Belgesel #%d] Sahne %d için medya seçilemedi.", state.doc_id, scene.index)
        return
    scene.approved_media = selected
    scene.color_palette = selected.get("dominant_colors", [])
    source_url = selected.get("source_url", "")
    if source_url:
        state.used_media_hashes.add(source_url[:64])


def _get_duration(path: Path) -> float:
    """ffprobe ile süre ölç."""
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


def _combine_clip_audio(
    clip_path: Path,
    audio_path: Path,
    output_path: Path,
) -> Path:
    """Video klip ile TTS sesini birleştir."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(clip_path),
            "-i", str(audio_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            str(output_path),
        ],
        check=True, capture_output=True,
    )
    return output_path


def _concat_scenes(state: DocumentaryState, work_dir: Path) -> Path:
    """
    FFmpeg concat demuxer ile tüm sahneleri birleştir.
    Her sahne: klip + tts ses birleştirilmiş.
    """
    combined_clips = []
    for scene in state.scenes:
        if not scene.final_clip_path or not scene.tts_path:
            log.warning(
                "Sahne %d eksik (clip=%s, tts=%s), atlanıyor.",
                scene.index, scene.final_clip_path, scene.tts_path,
            )
            continue

        clip_path = Path(scene.final_clip_path)
        tts_path = Path(scene.tts_path)
        combined_path = work_dir / f"combined_{scene.index:03d}.mp4"

        try:
            _combine_clip_audio(clip_path, tts_path, combined_path)
            combined_clips.append(combined_path)
            log.info("Sahne %d birleştirildi: %s", scene.index, combined_path.name)
        except subprocess.CalledProcessError as exc:
            log.error(
                "Sahne %d birleştirme hatası: %s",
                scene.index, exc.stderr.decode(errors="replace")[:300],
            )

    if not combined_clips:
        raise RuntimeError("Birleştirilecek tamamlanmış sahne bulunamadı.")

    # Concat listesi
    concat_file = work_dir / "concat_list.txt"
    concat_file.write_text(
        "\n".join(f"file '{p}'" for p in combined_clips),
        encoding="utf-8",
    )

    safe_title = re.sub(r"[^\w\-_\u0080-\uffff]", "_", state.title or state.topic)[:50]
    output_path = OUTPUT_DIR / f"{state.doc_id}_{safe_title}.mp4"

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_path),
        ],
        check=True, capture_output=True,
    )

    duration = _get_duration(output_path)
    log.info("Final video oluşturuldu: %s (%.1fs)", output_path.name, duration)
    return output_path


def _extract_json(raw: str) -> dict:
    """LLM çıktısından JSON çıkart."""
    # Doğrudan JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Herhangi bir { } bloğu
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    log.error("JSON çıkarılamadı. Ham çıktı (ilk 500): %s", raw[:500])
    raise ValueError("LLM çıktısından JSON çıkarılamadı.")


def run_documentary(topic: str, target_duration: int = 600) -> dict:
    """
    Tam belgesel üretim pipeline'ı.

    Args:
        topic: Belgesel konusu
        target_duration: Hedef süre (saniye, varsayılan 10 dakika)

    Returns:
        dict: {doc_id, title, output_path, scene_count, qa_score, status}
    """
    work_dir = Path(f"/tmp/curator_docs/{int(time.time())}")
    work_dir.mkdir(parents=True, exist_ok=True)
    log.info("Belgesel başlatılıyor: '%s' (dizin: %s)", topic, work_dir)

    doc_id = db.create_documentary(topic)
    state = DocumentaryState(doc_id=doc_id, topic=topic)

    try:
        # 1. Senaryo
        _update_status(state, "scripting")
        script_crew, _ = create_script_crew(topic, target_duration)
        script_result = script_crew.kickoff()
        script_data = _extract_json(script_result.raw)
        state = _apply_script(state, script_data)
        db.update_documentary_status(doc_id, "scripting", script_json=state.to_json())
        log.info("[#%d] Senaryo: '%s', %d sahne", doc_id, state.title, len(state.scenes))

        # 2. Her sahne için medya
        _update_status(state, "searching")
        for scene in state.scenes:
            if scene.index > 0:
                wait_secs = 70
                log.info("[#%d] Rate limit önlemi: %ds bekleniyor...", doc_id, wait_secs)
                time.sleep(wait_secs)
            log.info("[#%d] Sahne %d medya aranıyor...", doc_id, scene.index)
            try:
                media_crew = create_media_crew(scene, state)
                media_result = media_crew.kickoff()
                media_data = _extract_json(media_result.raw)
                _apply_media(state, scene, media_data)

                if scene.approved_media:
                    db.save_documentary_media(
                        documentary_id=doc_id,
                        scene_index=scene.index,
                        source=scene.approved_media.get("source", ""),
                        source_url=scene.approved_media.get("source_url", ""),
                        local_path=scene.approved_media.get("local_path", ""),
                        relevance_score=scene.approved_media.get("relevance_score", 0.0),
                    )
            except Exception as exc:
                log.error("[#%d] Sahne %d medya hatası: %s", doc_id, scene.index, exc)

        # 3. TTS + FFmpeg her sahne için
        _update_status(state, "assembling")
        tts_tool = KokoroTTSTool()
        ffmpeg_tool = FFmpegTool()

        for scene in state.scenes:
            # TTS
            tts_output = work_dir / f"tts_{scene.index:03d}.mp3"
            tts_result = json.loads(tts_tool._run(json.dumps({
                "text": scene.narration,
                "output_path": str(tts_output),
                "voice": "af_heart",
                "speed": 0.95,
            })))
            if tts_result.get("success"):
                scene.tts_path = tts_result["output_path"]
                log.info("[#%d] Sahne %d TTS: %.1fs", doc_id, scene.index, tts_result.get("duration_sec", 0))
            else:
                log.warning("[#%d] Sahne %d TTS başarısız.", doc_id, scene.index)

            # FFmpeg (medya varsa ve local_path dolu ise)
            clip_output = work_dir / f"clip_{scene.index:03d}.mp4"
            if scene.approved_media and scene.approved_media.get("local_path"):
                media = scene.approved_media
                ffmpeg_type = "ken_burns" if media.get("media_type") == "photo" else "clip"

                clip_result = json.loads(ffmpeg_tool._run(json.dumps({
                    "input_path": media.get("local_path", ""),
                    "output_path": str(clip_output),
                    "type": ffmpeg_type,
                    "duration": scene.duration_sec,
                    "clip_start": media.get("clip_start", 0.0),
                    "zoom_direction": "in" if scene.index % 2 == 0 else "out",
                })))

                if clip_result.get("success"):
                    scene.final_clip_path = clip_result["output_path"]
                    log.info("[#%d] Sahne %d klip hazır.", doc_id, scene.index)
                else:
                    log.warning("[#%d] Sahne %d FFmpeg başarısız.", doc_id, scene.index)
            else:
                log.warning(
                    "[#%d] Sahne %d için medya yok, siyah arka plan oluşturuluyor...",
                    doc_id, scene.index,
                )
                try:
                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-f", "lavfi",
                            "-i", f"color=black:size=1920x1080:duration={scene.duration_sec}:rate=25",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            str(clip_output),
                        ],
                        check=True, capture_output=True,
                    )
                    scene.final_clip_path = str(clip_output)
                except subprocess.CalledProcessError as exc:
                    log.error(
                        "[#%d] Sahne %d siyah klip hatası: %s",
                        doc_id, scene.index, exc.stderr.decode(errors="replace")[:300],
                    )

        # 4. Sahneleri birleştir
        output_path = _concat_scenes(state, work_dir)
        state.output_path = str(output_path)

        # 5. QA
        _update_status(state, "qa")
        qa_crew = create_qa_crew(state)
        qa_result = qa_crew.kickoff()
        try:
            qa_data = _extract_json(qa_result.raw)
        except Exception:
            qa_data = {"overall_score": 5.0, "approved": True, "technical_issues": []}

        qa_score = float(qa_data.get("overall_score", 5.0))
        log.info("[#%d] QA: skor=%.1f, onay=%s", doc_id, qa_score, qa_data.get("approved"))

        # 6. Tamamlandı
        _update_status(state, "done")
        db.update_documentary_status(doc_id, "done", output_path=str(output_path))

        return {
            "doc_id": doc_id,
            "title": state.title,
            "output_path": str(output_path),
            "scene_count": len(state.scenes),
            "qa_score": qa_score,
            "status": "done",
        }

    except Exception as exc:
        log.error("[#%d] Belgesel üretim hatası: %s", doc_id, exc, exc_info=True)
        db.update_documentary_status(doc_id, "error", error_msg=str(exc))
        raise

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        log.info("[#%d] Geçici dosyalar temizlendi: %s", doc_id, work_dir)


if __name__ == "__main__":
    import db as _db
    _db.init_db()
    print("✅ Orchestrator hazır.")
    print("   Kullanım: from documentary_system.orchestrator import run_documentary")
    print("   result = run_documentary('Osmanlı İmparatorluğu', target_duration=120)")
