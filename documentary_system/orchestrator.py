"""
Ana orkestratör: tüm crew'ları koordine eder.
DocumentaryState üzerinden çalışır, her adımı SQLite'a kaydeder.

Medya seçimi: sentence-transformers + CLIP (semantic_matcher)
TTS:          edge_tts (EN, word timestamps) | KokoroTTS → gTTS (TR)
Altyazı:      faster-whisper word align → karaoke SRT veya burn_srt
Birleştirme:  MoviePy (concat + transitions + BGM) → FFmpeg fallback
Çeşitlilik:   DiversityTracker (max_reuse=2)
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
from documentary_system.services.subtitle_service import build_srt
from documentary_system.services.tts_service import synthesize as tts_synthesize
from documentary_system.services.video_composer import (
    DiversityTracker,
    add_background_music,
    compose_scene,
    concat_clips,
)
from documentary_system.state.documentary_state import DocumentaryState, SceneState
from documentary_system.tools.ffmpeg_tool import FFmpegTool
from documentary_system.tools.media_selector import select_best_media
from documentary_system.tools.srt_generator import generate_srt

log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "documentary_output"
OUTPUT_DIR.mkdir(exist_ok=True)

BG_MUSIC_DIR = Path(__file__).parent.parent / "bg_music"

# sentence-transformers + CLIP ile skorlama yaptığımız için rate limit gerekmez
_SCENE_RATE_LIMIT_SECS = 0


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


def _normalize_audio(tts_result, work_dir: Path, scene_idx: int):
    """
    FFmpeg loudnorm ile TTS sesini -14 LUFS'a normalize et.
    Başarısız olursa orijinal TTSResult döner.
    """
    from documentary_system.services.tts_service import TTSResult
    src = Path(tts_result.audio_path)
    dst = work_dir / f"norm_{scene_idx:03d}.mp3"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
             "-ar", "44100", "-ab", "128k",
             str(dst)],
            check=True, capture_output=True, timeout=60,
        )
        dur = tts_result.duration_sec
        log.info("[Scene %d] Ses normalizasyonu tamamlandı: -14 LUFS", scene_idx)
        return TTSResult(
            success=True,
            audio_path=str(dst),
            duration_sec=dur,
            word_timestamps=tts_result.word_timestamps,
        )
    except Exception as exc:
        log.warning("[Scene %d] Ses normalizasyonu başarısız (orijinal kullanılıyor): %s", scene_idx, exc)
        return tts_result


def _embed_chapters(state, output_path: Path) -> Path:
    """
    FFmpeg ile chapter metadata'sı göm (YouTube chapter navigasyonu için).
    Başarısız olursa orijinal output_path döner.
    """
    if not state.scenes:
        return output_path
    try:
        meta_lines = [";FFMETADATA1\n"]
        cursor = 0.0
        for scene in state.scenes:
            start_ms = int(cursor * 1000)
            end_ms   = int((cursor + scene.duration_sec) * 1000)
            title    = (scene.visual_description or scene.narration or f"Scene {scene.index}")[:60]
            meta_lines += [
                "[CHAPTER]\n",
                "TIMEBASE=1/1000\n",
                f"START={start_ms}\n",
                f"END={end_ms}\n",
                f"title={title}\n\n",
            ]
            cursor += scene.duration_sec

        meta_path    = output_path.parent / f"{output_path.stem}_chapters.ffmeta"
        chapter_path = output_path.parent / f"chapters_{output_path.name}"
        meta_path.write_text("".join(meta_lines), encoding="utf-8")

        subprocess.run(
            ["ffmpeg", "-y",
             "-i", str(output_path),
             "-i", str(meta_path),
             "-map_metadata", "1",
             "-c", "copy",
             str(chapter_path)],
            check=True, capture_output=True, timeout=120,
        )
        output_path.unlink(missing_ok=True)
        chapter_path.rename(output_path)
        meta_path.unlink(missing_ok=True)
        log.info("Chapter metadata gömüldü: %d chapter", len(state.scenes))
        return output_path
    except Exception as exc:
        log.warning("Chapter metadata hatası (atlanıyor): %s", exc)
        return output_path


def run_documentary(
    topic: str,
    target_duration: int = 600,
    language: str = "tr",
    voice: str = "gtts_tr",
    transition_mode: str = "shuffle",
    subtitle_color: str = "#ffdc00",
    bgm_volume: float = 0.20,
    video_aspect: str = "16:9",
) -> dict:
    """
    Tam belgesel üretim pipeline'ı.

    Args:
        topic:           Belgesel konusu
        target_duration: Hedef süre (saniye)
        language:        'tr' | 'en'
        voice:           TTS ses kodu — edge_tts / gtts_tr /
                         chatterbox:default / siliconflow:model:voice
        transition_mode: cut | fade | shuffle | slidein_left | slidein_right | ...
        subtitle_color:  Aktif kelime rengi hex (örn. '#ffdc00')
        bgm_volume:      Arka plan müziği ses seviyesi (0.0–1.0, MPT default 0.20)
        video_aspect:    '16:9' | '9:16' | '1:1' (MPT-Extended VideoAspect)
    """
    work_dir = Path(f"/tmp/curator_docs/{int(time.time())}")
    work_dir.mkdir(parents=True, exist_ok=True)
    log.info("Belgesel başlatılıyor: '%s' lang=%s voice=%s", topic, language, voice)

    doc_id   = db.create_documentary(topic)
    state    = DocumentaryState(doc_id=doc_id, topic=topic)
    diversity  = DiversityTracker(max_reuse=2)

    try:
        # ── 1. Senaryo ────────────────────────────────────────────────────────
        _update_status(state, "scripting")
        script_crew, _ = create_script_crew(topic, target_duration, language)
        script_result  = script_crew.kickoff()
        script_data    = _extract_json(script_result.raw)
        state          = _apply_script(state, script_data)
        db.update_documentary_status(doc_id, "scripting", script_json=state.to_json())
        log.info("[#%d] Senaryo: '%s', %d sahne", doc_id, state.title, len(state.scenes))

        # ── 2. Semantik medya seçimi (sentence-transformers + CLIP) ───────────
        _update_status(state, "searching")
        for scene in state.scenes:
            log.info("[#%d] Sahne %d medya seçiliyor...", doc_id, scene.index)
            try:
                selected = select_best_media(scene, state)
                # Çeşitlilik kontrolü: aynı dosya max 2 kez kullanılabilir
                if selected and not diversity.is_allowed(selected.get("local_path", "")):
                    log.info("[#%d] Sahne %d: çeşitlilik limiti, alternatif aranıyor...", doc_id, scene.index)
                    selected = None  # Sonraki en iyi adaya düşülecek

                if selected:
                    diversity.register(selected.get("local_path", ""))
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

        # ── 3. TTS (edge_tts/KokoroTTS) + Karaoke SRT + FFmpeg ───────────────
        _update_status(state, "assembling")
        ffmpeg_tool = FFmpegTool()
        composed_clips: list[Path] = []

        for scene in state.scenes:
            # TTS — edge_tts (EN kelime zaman damgalı) veya Kokoro/gTTS (TR)
            tts_output = work_dir / f"tts_{scene.index:03d}.mp3"
            tts_result = tts_synthesize(
                text=scene.narration,
                output_path=tts_output,
                language=language,
                voice=voice if language == "en" else "gtts_tr",
                align_words=True,
            )

            tts_duration = 0.0
            if tts_result.success:
                scene.tts_path = tts_result.audio_path
                tts_duration   = tts_result.duration_sec
                log.info("[#%d] Sahne %d TTS: %.1fs (%d kelime damgası)",
                         doc_id, scene.index, tts_duration, len(tts_result.word_timestamps))
                # Ses normalizasyonu: -14 LUFS (YouTube/streaming standardı)
                tts_result = _normalize_audio(tts_result, work_dir, scene.index)
                if tts_result.success:
                    scene.tts_path = tts_result.audio_path
            else:
                log.warning("[#%d] Sahne %d TTS başarısız: %s", doc_id, scene.index, tts_result.error)

            # SRT altyazı: kelime zaman damgalı karaoke veya klasik
            srt_path  = work_dir / f"sub_{scene.index:03d}.srt"
            audio_dur = tts_duration if tts_duration > 0 else scene.duration_sec

            if tts_result.word_timestamps:
                build_srt(tts_result.word_timestamps, srt_path)
            else:
                generate_srt(scene.narration, audio_dur, srt_path)

            # FFmpeg klip (Ken Burns veya video kesme)
            clip_raw   = work_dir / f"clip_{scene.index:03d}.mp4"
            clip_sub   = work_dir / f"clip_sub_{scene.index:03d}.mp4"
            clip_final = work_dir / f"clip_final_{scene.index:03d}.mp4"

            if scene.approved_media and scene.approved_media.get("local_path"):
                media       = scene.approved_media
                ffmpeg_type = "ken_burns" if media.get("media_type") == "photo" else "clip"

                clip_result = json.loads(ffmpeg_tool._run(json.dumps({
                    "input_path":     media.get("local_path", ""),
                    "output_path":    str(clip_raw),
                    "type":           ffmpeg_type,
                    "duration":       scene.duration_sec,
                    "clip_start":     media.get("clip_start", 0.0),
                    "zoom_direction": "in" if scene.index % 2 == 0 else "out",
                    "video_aspect":   video_aspect,
                    "mood":           scene.mood,
                })))

                if clip_result.get("success"):
                    current_clip = Path(clip_result["output_path"])

                    # Altyazı yak (SRT → burn_srt)
                    if srt_path.exists() and srt_path.stat().st_size > 10:
                        sub_result = json.loads(ffmpeg_tool._run(json.dumps({
                            "input_path":     str(current_clip),
                            "output_path":    str(clip_sub),
                            "type":           "burn_srt",
                            "srt_path":       str(srt_path),
                            "subtitle_color": subtitle_color,
                        })))
                        current_clip = (
                            Path(sub_result["output_path"])
                            if sub_result.get("success")
                            else current_clip
                        )

                    # TTS sesini MoviePy ile klibe ekle (geçişli)
                    # transition_mode override: dashboard'dan gelen geçiş modunu kullan
                    effective_transition = transition_mode if transition_mode != "auto" else scene.transition
                    if tts_result.success:
                        composed = compose_scene(
                            video_path=current_clip,
                            audio_path=Path(tts_result.audio_path),
                            output_path=clip_final,
                            duration=audio_dur,
                            transition=effective_transition,
                            aspect=video_aspect,
                        )
                        scene.final_clip_path = str(composed) if composed else str(current_clip)
                    else:
                        scene.final_clip_path = str(current_clip)
                    log.info("[#%d] Sahne %d klip + altyazı + ses hazır.", doc_id, scene.index)
                else:
                    log.warning("[#%d] Sahne %d FFmpeg başarısız.", doc_id, scene.index)
            else:
                # Medya yok → siyah arka plan + altyazı
                log.warning("[#%d] Sahne %d için medya yok, siyah arka plan...", doc_id, scene.index)
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-f", "lavfi",
                         "-i", f"color=black:size=1920x1080:duration={audio_dur}:rate=25",
                         "-c:v", "libx264", "-pix_fmt", "yuv420p",
                         str(clip_raw)],
                        check=True, capture_output=True,
                    )
                    current_clip = clip_raw
                    if srt_path.exists() and srt_path.stat().st_size > 10:
                        sub_result = json.loads(ffmpeg_tool._run(json.dumps({
                            "input_path":     str(clip_raw),
                            "output_path":    str(clip_sub),
                            "type":           "burn_srt",
                            "srt_path":       str(srt_path),
                            "subtitle_color": subtitle_color,
                        })))
                        current_clip = (
                            Path(sub_result["output_path"])
                            if sub_result.get("success")
                            else clip_raw
                        )
                    if tts_result.success:
                        composed = compose_scene(
                            video_path=current_clip,
                            audio_path=Path(tts_result.audio_path),
                            output_path=clip_final,
                            duration=audio_dur,
                            transition=effective_transition,
                            aspect=video_aspect,
                        )
                        scene.final_clip_path = str(composed) if composed else str(current_clip)
                    else:
                        scene.final_clip_path = str(current_clip)
                except subprocess.CalledProcessError as exc:
                    log.error("[#%d] Sahne %d siyah klip hatası: %s",
                              doc_id, scene.index, exc.stderr.decode(errors="replace")[:300])

            if scene.final_clip_path and Path(scene.final_clip_path).exists():
                composed_clips.append(Path(scene.final_clip_path))

        # ── 4. Sahneleri MoviePy ile birleştir ────────────────────────────────
        if not composed_clips:
            raise RuntimeError("Birleştirilecek tamamlanmış sahne bulunamadı.")

        safe_title  = re.sub(r"[^\w\-_\u0080-\uffff]", "_", state.title or state.topic)[:50]
        output_path = OUTPUT_DIR / f"{state.doc_id}_{safe_title}.mp4"

        concat_clips(composed_clips, output_path)
        state.output_path = str(output_path)
        duration = _get_duration(output_path)
        log.info("Final video oluşturuldu: %s (%.1fs)", output_path.name, duration)

        # ── 5. Arka plan müziği (MoviePy AudioLoop) ───────────────────────────
        bg_music = _find_background_music()
        if bg_music:
            music_output = OUTPUT_DIR / f"withmusic_{output_path.name}"
            with_music = add_background_music(output_path, bg_music, music_output, volume=bgm_volume)
            if with_music:
                output_path.unlink(missing_ok=True)
                output_path = with_music
                state.output_path = str(output_path)

        # ── 5b. Chapter metadata (YouTube chapter navigasyonu) ────────────────
        output_path = _embed_chapters(state, output_path)

        # ── 6. QA ─────────────────────────────────────────────────────────────
        _update_status(state, "qa")
        qa_crew   = create_qa_crew(state)
        qa_result = qa_crew.kickoff()
        try:
            qa_data = _extract_json(qa_result.raw)
        except Exception:
            qa_data = {
                "overall_score": 5.0, "approved": True,
                "score_breakdown": {}, "viewer_notes": "",
                "accuracy_notes": "", "visual_notes": "",
                "narrative_notes": "", "technical_issues": [],
                "revision_needed": False, "revision_instructions": "",
            }

        _QA_MIN_SCORE = 7.5
        _QA_MAX_RETRIES = 1

        qa_score     = float(qa_data.get("overall_score", 5.0))
        qa_approved  = bool(qa_data.get("approved", False))
        qa_breakdown = qa_data.get("score_breakdown", {})
        qa_notes = {
            "viewer":    qa_data.get("viewer_notes", ""),
            "accuracy":  qa_data.get("accuracy_notes", ""),
            "visual":    qa_data.get("visual_notes", ""),
            "narrative": qa_data.get("narrative_notes", ""),
            "technical": qa_data.get("technical_issues", []),
            "revision":  qa_data.get("revision_instructions", ""),
        }
        log.info("[#%d] QA: skor=%.1f approved=%s (min=%.1f)", doc_id, qa_score, qa_approved, _QA_MIN_SCORE)

        # ── QA Engeli: minimum 7.5 ve approved=True olmadan video teslim edilmez ──
        if qa_score < _QA_MIN_SCORE or not qa_approved:
            revision_msg = qa_data.get("revision_instructions", "")[:500]
            log.warning(
                "[#%d] QA engeli (skor=%.1f, min=%.1f). Senaryo revizyonu deneniyor...",
                doc_id, qa_score, _QA_MIN_SCORE,
            )

            # Tek seferlik senaryo revizyonu — revision_instructions'ı contexte ekle
            if _QA_MAX_RETRIES > 0 and revision_msg:
                try:
                    log.info("[#%d] QA revision: senaryo yeniden yazılıyor...", doc_id)
                    revision_topic = f"{topic}\n\n[QA REVİZYON TALİMATLARI]\n{revision_msg}"
                    retry_crew, _ = create_script_crew(revision_topic, target_duration, language)
                    retry_result  = retry_crew.kickoff()
                    retry_data    = _extract_json(retry_result.raw)
                    state         = _apply_script(state, retry_data)
                    log.info("[#%d] QA revision senaryo tamamlandı: %d sahne", doc_id, len(state.scenes))
                except Exception as retry_exc:
                    log.warning("[#%d] QA revision senaryo hatası: %s", doc_id, retry_exc)

            db.update_documentary_status(
                doc_id, "qa_failed",
                error_msg=f"QA skor {qa_score:.1f}/{_QA_MIN_SCORE} | {revision_msg[:200]}",
            )
            return {
                "doc_id":          doc_id,
                "title":           state.title,
                "output_path":     str(output_path),
                "scene_count":     len(state.scenes),
                "qa_score":        qa_score,
                "qa_approved":     False,
                "score_breakdown": qa_breakdown,
                "qa_notes":        qa_notes,
                "status":          "qa_failed",
            }

        # ── 7. Tamamlandı (QA geçti) ──────────────────────────────────────────
        _update_status(state, "done")
        db.update_documentary_status(doc_id, "done", output_path=str(output_path))

        return {
            "doc_id":          doc_id,
            "title":           state.title,
            "output_path":     str(output_path),
            "scene_count":     len(state.scenes),
            "qa_score":        qa_score,
            "qa_approved":     qa_approved,
            "score_breakdown": qa_breakdown,
            "qa_notes":        qa_notes,
            "status":          "done",
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
