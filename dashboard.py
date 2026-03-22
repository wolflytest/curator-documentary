"""
Streamlit Web Dashboard — Belgesel Stüdyosu
Çalıştır: streamlit run dashboard.py --server.port 8501
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

# Proje path'ini ekle
sys.path.insert(0, str(Path(__file__).parent))

import db
import config

db.init_db()

# ── SAYFA AYARLARI ──────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Belgesel Stüdyosu",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎬 Belgesel Stüdyosu")

# ── SIDEBAR: AYARLAR ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Ayarlar")

    # Dil seçimi
    language = st.selectbox(
        "🌍 Dil",
        options=["tr", "en"],
        format_func=lambda x: "🇹🇷 Türkçe" if x == "tr" else "🇬🇧 İngilizce",
        index=0,
    )

    # Dile göre ses seçimi
    voices = config.TTS_VOICES.get(language, {})
    voice_keys   = list(voices.keys())
    voice_labels = list(voices.values())

    voice_idx = st.selectbox(
        "🎤 Ses",
        options=range(len(voice_keys)),
        format_func=lambda i: voice_labels[i],
        index=0,
    )
    selected_voice = voice_keys[voice_idx]

    # Ses örneği linki
    if language == "en":
        st.markdown(
            "💡 [Sesleri Dinle →](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)",
            unsafe_allow_html=True,
        )

    # Video süresi
    duration_options = {int(k): v for k, v in config.VIDEO_DURATIONS.items()}
    duration = st.select_slider(
        "⏱ Video Süresi",
        options=sorted(duration_options.keys()),
        value=600,
        format_func=lambda x: duration_options[x],
    )

    # Aspect ratio (MPT-Extended VideoAspect)
    aspect_options = config.VIDEO_ASPECTS
    aspect_ratio = st.selectbox(
        "📐 Video Formatı",
        options=list(aspect_options.keys()),
        format_func=lambda x: aspect_options[x],
        index=0,
    )

    st.divider()
    st.write("🎬 Geçiş & Efektler")

    # Geçiş modu (MPT-Extended'dan)
    transition_options = {
        "shuffle":        "🔀 Rastgele (Shuffle)",
        "cut":            "✂️ Direkt Kesim (Cut)",
        "fade":           "🌅 Kararma (Fade)",
        "slidein_left":   "⬅️ Soldan Kaydırma",
        "slidein_right":  "➡️ Sağdan Kaydırma",
        "slidein_top":    "⬆️ Yukarıdan Kaydırma",
        "slidein_bottom": "⬇️ Aşağıdan Kaydırma",
    }
    transition_mode = st.selectbox(
        "🎞️ Geçiş Modu",
        options=list(transition_options.keys()),
        format_func=lambda x: transition_options[x],
        index=0,
    )

    # Altyazı rengi (MPT-Extended'dan)
    subtitle_color = st.color_picker(
        "🎨 Altyazı Vurgu Rengi", value="#ffdc00"
    )

    # BGM ses seviyesi (MPT-Extended default %20)
    bgm_volume = st.slider(
        "🎵 Müzik Sesi", min_value=0, max_value=50, value=20, step=5,
        format="%d%%",
    ) / 100.0

    st.divider()
    st.caption("Oracle Cloud ARM üzerinde çalışıyor")

# ── ANA ALAN: SEKMELER ───────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎬 Yeni Belgesel", "📋 Geçmiş"])

# ── SEKME 1: YENİ BELGESEL ──────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        topic = st.text_input(
            "📌 Belgesel Konusu",
            placeholder="Örn: Osmanlı İmparatorluğu'nun Yükselişi",
        )

    with col2:
        st.write("")
        st.write("")
        start_btn = st.button("🎬 Üretimi Başlat", type="primary", use_container_width=True)

    # Ayar özeti
    lang_emoji  = "🇹🇷" if language == "tr" else "🇬🇧"
    voice_label = voices.get(selected_voice, selected_voice)
    dur_label   = duration_options[duration]

    st.info(
        f"{lang_emoji} **{language.upper()}** · 🎤 {voice_label} · ⏱ {dur_label} · "
        f"📐 {aspect_ratio} · "
        f"🎞️ {transition_options[transition_mode].split(' ')[1]} · 🎵 {int(bgm_volume*100)}%",
        icon="ℹ️",
    )

    if start_btn:
        if not topic.strip():
            st.error("❌ Lütfen bir konu girin.")
        else:
            st.divider()
            progress_area = st.empty()
            log_area      = st.empty()

            with st.spinner(f"🎬 '{topic}' belgesi üretiliyor..."):
                progress_area.progress(0, text="Başlatılıyor...")

                try:
                    # Orchestrator'ı subprocess olarak çalıştır
                    # (Streamlit'in event loop'u ile çakışmasın)
                    cmd = [
                        sys.executable, "-c",
                        f"""
import sys; sys.path.insert(0, '.')
import db, json
db.init_db()
from documentary_system.orchestrator import run_documentary
result = run_documentary(
    topic={repr(topic.strip())},
    target_duration={duration},
    language={repr(language)},
    voice={repr(selected_voice)},
    transition_mode={repr(transition_mode)},
    subtitle_color={repr(subtitle_color)},
    bgm_volume={bgm_volume},
    video_aspect={repr(aspect_ratio)},
)
print(json.dumps(result, ensure_ascii=False))
""",
                    ]

                    progress_area.progress(10, text="⚙️ Sistem hazırlanıyor...")

                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=str(Path(__file__).parent),
                    )

                    progress_area.progress(20, text="📝 Senaryo yazılıyor...")

                    # Sonucu bekle (timeout: 30 dakika)
                    stdout, stderr = process.communicate(timeout=1800)

                    if process.returncode == 0:
                        # Son satır JSON sonucu
                        lines       = stdout.strip().split("\n")
                        result_json = lines[-1]
                        result      = json.loads(result_json)

                        progress_area.progress(100, text="✅ Tamamlandı!")

                        st.success("🎉 Belgesel hazır!")

                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("🎬 Sahne",    result["scene_count"])
                        col_b.metric("⭐ QA Skoru", f"{result['qa_score']:.1f}/10")
                        col_c.metric("📁 Durum",    result["status"].upper())

                        # QA skor rengi
                        qa_score = result.get("qa_score", 0)
                        if qa_score >= 9.0:
                            st.success(f"✅ QA Skoru: {qa_score:.1f}/10 — Yayın Kalitesi")
                        elif qa_score >= 7.0:
                            st.warning(f"⚠️ QA Skoru: {qa_score:.1f}/10 — İyileştirme Önerildi")
                        else:
                            st.error(f"❌ QA Skoru: {qa_score:.1f}/10 — Revizyon Gerekli")

                        # Skor breakdown
                        breakdown = result.get("score_breakdown", {})
                        if breakdown:
                            st.write("**📊 Skor Detayı:**")
                            bd_cols = st.columns(5)
                            labels = [
                                ("🎭 Engagement", "engagement"),
                                ("✅ Doğruluk", "accuracy"),
                                ("🎬 Görsel", "visual_sync"),
                                ("✍️ Anlatım", "narrative"),
                                ("⚙️ Teknik", "technical"),
                            ]
                            for col, (label, key) in zip(bd_cols, labels):
                                val = breakdown.get(key, "-")
                                col.metric(label, f"{val:.1f}" if isinstance(val, float) else val)

                        # QA notları
                        qa_notes = result.get("qa_notes", {})
                        if any(qa_notes.values()):
                            with st.expander("📝 QA Notları — Neden Bu Skoru Aldı?"):
                                if qa_notes.get("viewer"):
                                    st.write("**🎭 İzleyici Deneyimi:**", qa_notes["viewer"])
                                if qa_notes.get("accuracy"):
                                    st.write("**✅ Doğruluk:**", qa_notes["accuracy"])
                                if qa_notes.get("visual"):
                                    st.write("**🎬 Görsel Uyum:**", qa_notes["visual"])
                                if qa_notes.get("narrative"):
                                    st.write("**✍️ Anlatım:**", qa_notes["narrative"])
                                if qa_notes.get("technical"):
                                    st.write("**⚙️ Teknik Sorunlar:**")
                                    for issue in qa_notes["technical"]:
                                        st.write(f"  - {issue}")
                                if qa_notes.get("revision"):
                                    st.info(f"💡 **Revizyon Önerileri:**\n{qa_notes['revision']}")

                        st.subheader(f"📹 {result['title']}")

                        output_path = Path(result["output_path"])
                        if output_path.exists():
                            st.video(str(output_path))
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    label="⬇️ MP4 İndir",
                                    data=f,
                                    file_name=output_path.name,
                                    mime="video/mp4",
                                    type="primary",
                                )
                        else:
                            st.warning(f"Video dosyası bulunamadı: {output_path}")
                    else:
                        progress_area.empty()
                        st.error("❌ Üretim hatası!")
                        with st.expander("Hata detayları"):
                            st.code(stderr[-3000:] if stderr else "Bilinmeyen hata")

                except subprocess.TimeoutExpired:
                    process.kill()
                    st.error("⏱ Zaman aşımı — işlem 30 dakikayı geçti.")
                except Exception as exc:
                    st.error(f"❌ Hata: {exc}")

# ── SEKME 2: GEÇMİŞ ─────────────────────────────────────────────────
with tab2:
    st.subheader("📋 Belgesel Geçmişi")

    if st.button("🔄 Yenile"):
        st.rerun()

    rows = db.list_documentaries(limit=50)

    if not rows:
        st.info("📭 Henüz belgesel üretilmemiş.")
    else:
        status_icons = {
            "done": "✅", "error": "❌", "pending": "⏳",
            "scripting": "📝", "searching": "🔍", "assembling": "🎞️", "qa": "🔎",
        }
        for row in rows:
            icon = status_icons.get(row["status"], "🔄")
            with st.expander(f"{icon} [{row['id']}] {row['topic']} — {row['created_at'][:16]}"):
                col1, col2 = st.columns(2)
                col1.write(f"**Durum:** {row['status']}")
                col2.write(f"**Tarih:** {row['created_at'][:16]}")

                if row.get("output_path"):
                    output_path = Path(row["output_path"])
                    if output_path.exists():
                        st.video(str(output_path))
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="⬇️ İndir",
                                data=f,
                                file_name=output_path.name,
                                mime="video/mp4",
                                key=f"dl_{row['id']}",
                            )
                    else:
                        st.warning("Video dosyası bulunamadı.")

                if row.get("error_msg"):
                    st.error(f"Hata: {row['error_msg'][:200]}")
