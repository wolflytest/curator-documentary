"""
Streamlit web arayüzü: belgesel üretimi ve geçmiş görüntüleme.
Çalıştır: streamlit run dashboard.py
"""
import json
import subprocess
import threading
import time
from pathlib import Path

import streamlit as st

import db
from config import TTS_VOICES, VIDEO_DURATIONS
from documentary_system import orchestrator

# ── Sayfa ayarları ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Küratör — Belgesel Stüdyosu",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

db.init_db()

# ── Yan menü ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 Belgesel Stüdyosu")
    page = st.radio(
        "Sayfa",
        ["Yeni Belgesel", "Geçmiş", "İstatistikler"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Küratör v1.0 · CrewAI + Kokoro TTS")


# ── YENİ BELGESEL ─────────────────────────────────────────────────────────────
if page == "Yeni Belgesel":
    st.header("🎬 Yeni Belgesel Üret")

    col1, col2 = st.columns([2, 1])

    with col1:
        topic = st.text_input(
            "Belgesel Konusu",
            placeholder="Örn: Osmanlı İmparatorluğu'nun Yükselişi",
            help="Belgesel konusunu giriniz. Ne kadar spesifik, o kadar iyi.",
        )

    with col2:
        language = st.selectbox(
            "Anlatım Dili",
            options=["tr", "en"],
            format_func=lambda x: "🇹🇷 Türkçe" if x == "tr" else "🇬🇧 İngilizce",
        )

    col3, col4 = st.columns(2)
    with col3:
        voice_label = st.selectbox("Ses", options=list(TTS_VOICES.keys()))
        voice_code = TTS_VOICES[voice_label]

    with col4:
        dur_label = st.selectbox("Hedef Süre", options=list(VIDEO_DURATIONS.keys()))
        duration_secs = VIDEO_DURATIONS[dur_label]

    st.divider()

    if st.button("🚀 Belgesel Üret", type="primary", disabled=not topic.strip()):
        if not topic.strip():
            st.error("Lütfen bir konu girin.")
        else:
            progress_bar = st.progress(0, text="Başlatılıyor...")
            status_box   = st.empty()
            result_box   = st.empty()

            stages = [
                (15, "scripting",   "📝 Senaryo yazılıyor..."),
                (40, "searching",   "🔍 Medya aranıyor..."),
                (70, "assembling",  "🎞 Sahneler birleştiriliyor..."),
                (90, "qa",          "✅ Kalite kontrolü..."),
                (100, "done",       "🎉 Tamamlandı!"),
            ]

            _result: dict = {}
            _error:  list = []

            def _run() -> None:
                try:
                    _result.update(orchestrator.run_documentary(
                        topic.strip(), duration_secs, language, voice_code
                    ))
                except Exception as exc:  # noqa: BLE001
                    _error.append(str(exc))

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()

            stage_idx = 0
            while thread.is_alive():
                if stage_idx < len(stages) - 1:
                    pct, stage_key, stage_msg = stages[stage_idx]
                    # Gerçek DB durumunu kontrol et
                    docs = db.list_documentaries(limit=1)
                    if docs:
                        current_status = docs[0].get("status", "")
                        for i, (p, k, m) in enumerate(stages):
                            if k == current_status and i >= stage_idx:
                                stage_idx = i
                                break
                    pct, _, stage_msg = stages[stage_idx]
                    progress_bar.progress(pct, text=stage_msg)
                time.sleep(3)

            thread.join()

            if _error:
                progress_bar.empty()
                st.error(f"❌ Hata: {_error[0]}")
            elif _result:
                progress_bar.progress(100, text="🎉 Tamamlandı!")
                out_path = Path(_result.get("output_path", ""))
                st.success(f"✅ **{_result['title']}** oluşturuldu!")

                info_col1, info_col2, info_col3 = st.columns(3)
                info_col1.metric("Sahne Sayısı", _result.get("scene_count", "—"))
                info_col2.metric("QA Skoru", f"{_result.get('qa_score', 0):.1f}/10")
                info_col3.metric("Süre (hedef)", dur_label)

                if out_path.exists():
                    st.video(str(out_path))
                    with open(out_path, "rb") as f:
                        st.download_button(
                            "⬇️ MP4 İndir",
                            data=f,
                            file_name=out_path.name,
                            mime="video/mp4",
                        )
                else:
                    st.info(f"Dosya: `{out_path}`")


# ── GEÇMİŞ ───────────────────────────────────────────────────────────────────
elif page == "Geçmiş":
    st.header("📚 Belgesel Geçmişi")

    docs = db.list_documentaries(limit=50)
    if not docs:
        st.info("Henüz belgesel üretilmemiş.")
    else:
        status_icons = {"done": "✅", "error": "❌", "pending": "⏳", "scripting": "📝",
                        "searching": "🔍", "assembling": "🎞", "qa": "🔬"}

        for doc in docs:
            icon = status_icons.get(doc["status"], "🔄")
            with st.expander(f"{icon} [{doc['id']}] {doc['topic']} — {doc['created_at'][:16]}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Durum:** {doc['status']}")
                    if doc.get("error_msg"):
                        st.error(f"Hata: {doc['error_msg']}")
                    if doc.get("output_path"):
                        out_path = Path(doc["output_path"])
                        if out_path.exists():
                            st.video(str(out_path))
                            with open(out_path, "rb") as f:
                                st.download_button(
                                    "⬇️ İndir",
                                    data=f,
                                    file_name=out_path.name,
                                    mime="video/mp4",
                                    key=f"dl_{doc['id']}",
                                )
                        else:
                            st.caption(f"Dosya mevcut değil: `{out_path}`")
                with col2:
                    st.caption(f"ID: {doc['id']}")
                    st.caption(f"Oluşturuldu: {doc['created_at'][:19]}")

                # Script detayları
                if doc.get("script_json"):
                    try:
                        script = json.loads(doc["script_json"])
                        if st.checkbox("Senaryo göster", key=f"sc_{doc['id']}"):
                            scenes = script.get("scenes", [])
                            for scene in scenes:
                                st.markdown(
                                    f"**Sahne {scene.get('index', '?')}** ({scene.get('duration_sec', 0):.0f}s) "
                                    f"— _{scene.get('mood', '')}_\n\n"
                                    f"{scene.get('narration', '')}"
                                )
                                st.divider()
                    except (json.JSONDecodeError, TypeError):
                        pass


# ── İSTATİSTİKLER ─────────────────────────────────────────────────────────────
elif page == "İstatistikler":
    st.header("📊 İstatistikler")

    docs = db.list_documentaries(limit=200)
    contents = db.get_all_contents(limit=200)

    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Belgesel", len(docs))
    col2.metric("Tamamlanan", sum(1 for d in docs if d["status"] == "done"))
    col3.metric("Kaydedilen İçerik", len(contents))

    st.divider()
    st.subheader("Belgesel Durumu Dağılımı")

    from collections import Counter
    status_counts = Counter(d["status"] for d in docs)
    if status_counts:
        import pandas as pd
        df_status = pd.DataFrame(
            {"Durum": list(status_counts.keys()), "Sayı": list(status_counts.values())}
        )
        st.bar_chart(df_status.set_index("Durum"))

    st.subheader("İçerik Platform Dağılımı")
    stats = db.get_stats()
    if stats["platforms"]:
        import pandas as pd
        df_plat = pd.DataFrame(stats["platforms"], columns=["Platform", "Sayı"])
        st.bar_chart(df_plat.set_index("Platform"))

    if stats["top_tags"]:
        st.subheader("En Çok Kullanılan Etiketler")
        import pandas as pd
        df_tags = pd.DataFrame(stats["top_tags"], columns=["Etiket", "Sayı"])
        st.dataframe(df_tags, use_container_width=True, hide_index=True)
