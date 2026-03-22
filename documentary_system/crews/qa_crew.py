"""
Final kalite kontrol ekibi — 5 ajan, minimum skor 9.0.

Ajanlar:
  1. İzleyici Deneyimi Analisti — engagement ve retention
  2. İçerik Doğruluk Denetçisi — tarihi/bilimsel doğruluk
  3. Görsel-Ses Uyum Uzmanı — medya-narrasyon uyumu
  4. Anlatım Akışı Editörü — senaryo kalitesi ve dil
  5. Teknik Kalite Denetçisi — video/ses teknik kalite + final karar
"""
import logging

from crewai import Agent, Crew, Process, Task

from documentary_system.llm_config import get_llm
from documentary_system.state.documentary_state import DocumentaryState

log = logging.getLogger(__name__)

_MIN_SCORE = 9.0  # Minimum kabul edilebilir QA skoru


def create_qa_crew(state: DocumentaryState) -> Crew:
    """5 ajanlı QA ekibini oluştur. Minimum skor 9.0."""
    llm = get_llm()

    # Ortak bağlam bilgisi
    _ctx = (
        f"Belgesel Başlık: {state.title}\n"
        f"Konu: {state.topic}\n"
        f"Sahne sayısı: {len(state.scenes)}\n"
        f"Video dosyası: {state.output_path}\n"
    )

    # ── AGENT 1: İzleyici Deneyimi Analisti ─────────────────────────────────
    viewer = Agent(
        role="İzleyici Deneyimi ve Engagement Analisti",
        goal=(
            "Videoyu gerçek bir izleyici olarak değerlendir. "
            "Hook gücü, retention, merak uyandırma ve duygusal bağlantıyı ölç."
        ),
        backstory=(
            "Netflix ve YouTube içerik stratejisti, 8 yıl deneyim. "
            "İzleyici davranış verilerini analiz eder. "
            "İlk 10 saniye hook yoksa izleyicinin %80'i ayrılır — bunu bilir. "
            "Yapay zeka üretimi içerikleri insan yapımından ayırt eder."
        ),
        llm=llm,
        verbose=True,
        max_iter=2,
    )

    # ── AGENT 2: İçerik Doğruluk Denetçisi ──────────────────────────────────
    fact_checker = Agent(
        role="İçerik Doğruluk ve Güvenilirlik Denetçisi",
        goal=(
            "Senaryodaki tarihi/bilimsel iddiaları doğrula. "
            "Yanlış tarihler, yanıltıcı genellemeler ve olgusal hatalar bul."
        ),
        backstory=(
            "Oxford tarih doktoru, 12 yıl akademik araştırma. "
            "Her iddia kanıtlanabilir olmalı. "
            "Yapay zeka halüsinasyonlarını tespit etme konusunda deneyimli. "
            "Yanlış bir tarihsel iddia skor'u 3 puan düşürür."
        ),
        llm=llm,
        verbose=True,
        max_iter=2,
    )

    # ── AGENT 3: Görsel-Ses Uyum Uzmanı ─────────────────────────────────────
    visual_sync = Agent(
        role="Görsel-Ses Uyum ve Medya Kalite Uzmanı",
        goal=(
            "Her sahnede görsel ile narrasyon uyumunu değerlendir. "
            "Alakasız görsel, kötü geçiş, renk/ton tutarsızlıklarını tespit et."
        ),
        backstory=(
            "BBC ve NatGeo için 10 yıl post-prodüksiyon direktörü. "
            "Anlatılan ile gösterilen örtüşmeli. "
            "Stok fotoğraf kalitesini, telif hakkı riskini, görsel çeşitliliği kontrol eder."
        ),
        llm=llm,
        verbose=True,
        max_iter=2,
    )

    # ── AGENT 4: Anlatım Akışı ve Dil Editörü ───────────────────────────────
    narrative_editor = Agent(
        role="Anlatım Akışı ve Senaryo Kalite Editörü",
        goal=(
            "Senaryonun dil kalitesini, akışını ve dramatik yapısını değerlendir. "
            "Tekrarlayan ifadeler, zayıf geçişler ve monoton anlatımı tespit et."
        ),
        backstory=(
            "20 yıl belgesel senaryo editörü. "
            "Her cümle öncekinden daha ilgi çekici olmalı. "
            "Pasif çatı, monoton ritim ve klişe ifadeler -2 puan getirir. "
            "BBC World Service kalite standartlarını uygular."
        ),
        llm=llm,
        verbose=True,
        max_iter=2,
    )

    # ── AGENT 5: Teknik Kalite Denetçisi + Final Karar ───────────────────────
    tech_qa = Agent(
        role="Teknik Kalite Denetçisi ve QA Karar Mercii",
        goal=(
            "Teknik kaliteyi kontrol et ve tüm ajan raporlarını sentezleyerek "
            "final QA kararını ver. Minimum skor 9.0 gerekli."
        ),
        backstory=(
            "Netflix teknik standartları sertifikalı post-prodüksiyon mühendisi. "
            "Ses seviyesi -14 LUFS, video CRF 18, altyazı hizalaması kontrol eder. "
            "4 ajanın raporunu değerlendirip adil ağırlıklı final skor hesaplar: "
            "engagement %30, doğruluk %25, görsel uyum %25, dil kalitesi %20. "
            "Skor 9.0 altıysa açık revision_instructions yazar."
        ),
        llm=llm,
        verbose=True,
        max_iter=3,
    )

    # ── TASK 1: İzleyici Değerlendirmesi ────────────────────────────────────
    viewer_task = Task(
        description=(
            f"{_ctx}\n"
            "İzleyici deneyimi değerlendirmesi:\n"
            "1. İlk 10 saniye hook gücü (1-10)\n"
            "2. Merak uyandırma ve duygusal bağlantı (1-10)\n"
            "3. Monotonluk ve sıkılma noktaları — hangi sahneler?\n"
            "4. 'Bunu paylaşır mıyım?' sorusuna cevap\n"
            "5. Engagement skoru (1-10)\n"
            "Her puana neden o puanı verdiğini açıkla."
        ),
        expected_output="Engagement analiz raporu: hook skoru, duygusal etki, monotonluk noktaları, genel engagement skoru.",
        agent=viewer,
    )

    # ── TASK 2: Gerçek Doğruluk Kontrolü ────────────────────────────────────
    fact_task = Task(
        description=(
            f"{_ctx}\n"
            "İçerik doğruluk kontrolü:\n"
            "1. Senaryodaki her önemli tarihi/bilimsel iddiayı listele\n"
            "2. Doğrulanamayanları veya yanlış olanları işaretle\n"
            "3. Yapay zeka halüsinasyonu riski var mı?\n"
            "4. Doğruluk skoru (1-10)\n"
            "Sahne index'leriyle birlikte sorunlu noktaları listele."
        ),
        expected_output="Doğruluk raporu: doğrulanan iddialar, şüpheli iddialar, halüsinasyon riski, doğruluk skoru.",
        agent=fact_checker,
    )

    # ── TASK 3: Görsel-Ses Uyum Kontrolü ────────────────────────────────────
    visual_task = Task(
        description=(
            f"{_ctx}\n"
            "Görsel-ses uyum değerlendirmesi:\n"
            "1. Sahnelerde görsel-narrasyon uyumu güçlü mü?\n"
            "2. Geçişler dramatik tona uygun mu?\n"
            "3. Görsel çeşitlilik yeterli mi (aynı stok görsel tekrar var mı)?\n"
            "4. Görsel uyum skoru (1-10)\n"
            "Zayıf sahneleri index ile belirt."
        ),
        expected_output="Görsel uyum raporu: uyumlu sahneler, sorunlu sahneler, görsel çeşitlilik analizi, uyum skoru.",
        agent=visual_sync,
        context=[viewer_task],
    )

    # ── TASK 4: Anlatım Kalitesi Kontrolü ───────────────────────────────────
    narrative_task = Task(
        description=(
            f"{_ctx}\n"
            "Anlatım kalitesi değerlendirmesi:\n"
            "1. Dil akıcılığı ve zenginliği (1-10)\n"
            "2. Tekrar eden ifadeler veya klişeler var mı?\n"
            "3. Sahne geçişleri mantıksal akış sağlıyor mu?\n"
            "4. BBC/NatGeo kalite standardını karşılıyor mu?\n"
            "5. Anlatım skoru (1-10)\n"
            "Zayıf cümleleri veya sahneleri örnekle."
        ),
        expected_output="Anlatım kalitesi raporu: dil zenginliği, tekrarlar, akış analizi, anlatım skoru.",
        agent=narrative_editor,
        context=[fact_task],
    )

    # ── TASK 5: Final QA Kararı ──────────────────────────────────────────────
    tech_task = Task(
        description=(
            f"{_ctx}\n"
            "Teknik kontroller:\n"
            "1. Ses seviyesi tutarlı mı? (-14 LUFS hedef)\n"
            "2. Video 1080p, CRF 18 kalitede mi?\n"
            "3. Altyazı hizalaması doğru mu?\n"
            "4. Geçişler pürüzsüz mü?\n\n"
            "Tüm ajan raporlarını (engagement, doğruluk, görsel, anlatım) değerlendirerek "
            "ağırlıklı final skor hesapla:\n"
            "  engagement_score × 0.30\n"
            "  accuracy_score × 0.25\n"
            "  visual_score × 0.25\n"
            "  narrative_score × 0.20\n\n"
            f"MİNİMUM KABUL SKORU: {_MIN_SCORE}/10\n"
            "Skor bu altındaysa approved=false ve revision_instructions'a SOMUT öneriler yaz.\n\n"
            "SADECE şu JSON formatında yanıt ver:\n"
            "{\n"
            '  "overall_score": 8.5,\n'
            '  "approved": false,\n'
            '  "score_breakdown": {\n'
            '    "engagement": 8.0,\n'
            '    "accuracy": 9.0,\n'
            '    "visual_sync": 8.5,\n'
            '    "narrative": 8.0,\n'
            '    "technical": 9.0\n'
            '  },\n'
            '  "viewer_notes": "İzleyici deneyimi özeti",\n'
            '  "accuracy_notes": "Doğruluk sorunları",\n'
            '  "visual_notes": "Görsel uyum sorunları",\n'
            '  "narrative_notes": "Anlatım sorunları",\n'
            '  "technical_issues": ["sorun1", "sorun2"],\n'
            '  "revision_needed": true,\n'
            '  "revision_instructions": "1. Hook güçlendir\\n2. Sahne 3 görseli değiştir"\n'
            "}"
        ),
        expected_output="JSON formatında kapsamlı QA raporu",
        agent=tech_qa,
        context=[viewer_task, fact_task, visual_task, narrative_task],
    )

    return Crew(
        agents=[viewer, fact_checker, visual_sync, narrative_editor, tech_qa],
        tasks=[viewer_task, fact_task, visual_task, narrative_task, tech_task],
        process=Process.sequential,
        verbose=True,
    )


if __name__ == "__main__":
    from documentary_system.state.documentary_state import DocumentaryState
    state = DocumentaryState(doc_id=1, topic="Test")
    state.title = "Test Belgeseli"
    state.output_path = "/tmp/test.mp4"
    crew = create_qa_crew(state)
    print("✅ QA crew oluşturuldu")
    print(f"   Ajanlar: {[a.role for a in crew.agents]}")
