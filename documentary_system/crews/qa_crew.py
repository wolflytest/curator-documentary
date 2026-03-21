"""
Final kalite kontrol ekibi.
Video tamamlandıktan sonra çalışır.
"""
import logging

from crewai import Agent, Crew, Process, Task

from documentary_system.llm_config import get_llm
from documentary_system.state.documentary_state import DocumentaryState

log = logging.getLogger(__name__)


def create_qa_crew(state: DocumentaryState) -> Crew:
    """QA ekibini oluştur. Video kalitesini değerlendirir."""
    llm = get_llm()

    # AGENT 1: Ortalama YouTube İzleyicisi
    viewer = Agent(
        role="Ortalama YouTube İzleyicisi",
        goal=(
            "Videoyu sıradan bir izleyici gözüyle değerlendir, "
            "nerede sıkıldığını, nerede ilginin düştüğünü raporla."
        ),
        backstory=(
            "25 yaşında, günde 2 saat YouTube izleyen biri. "
            "İlk 30 saniyede ilgini çekmeyen videoyu kapatırsın. "
            "Tarihi içerik sever ama ders gibi anlatım olmayan. "
            "Görsel-ses uyumsuzluğunu hemen fark edersin."
        ),
        llm=llm,
        verbose=True,
        max_iter=2,
    )

    # AGENT 2: Teknik Kalite Denetçisi
    tech_qa = Agent(
        role="Post-Prodüksiyon Teknik Denetçisi",
        goal=(
            "Videonun teknik kalitesini kontrol et: "
            "ses seviyesi, görüntü geçişleri, altyazı hataları. "
            "Final JSON kararını ver."
        ),
        backstory=(
            "Prodüksiyon şirketinde 10 yıl. Ses patlaması, "
            "geçiş hatası, yanlış hizalanmış altyazı, çözünürlük sorunu — "
            "hepsini fark eder. Çıktı raporunda tam timestamp ve sorun tanımı verir."
        ),
        llm=llm,
        verbose=True,
        max_iter=2,
    )

    # TASK 1: İzleyici Değerlendirmesi
    viewer_task = Task(
        description=(
            f"Belgesel değerlendirmesi:\n"
            f"Başlık: {state.title}\n"
            f"Sahne sayısı: {len(state.scenes)}\n"
            f"Video: {state.output_path}\n\n"
            "Ortalama bir YouTube izleyicisi olarak değerlendir:\n"
            "1. İlk 30 saniye seni tutuyor mu?\n"
            "2. Anlatım ilgi çekici mi, sıkıcı mı?\n"
            "3. Görsel-ses uyumu nasıl?\n"
            "4. Nerede izlemeyi bırakırsın?\n"
            "5. Genel izleme deneyimi 1-10 arası?"
        ),
        expected_output="İzleyici perspektifinden detaylı değerlendirme raporu.",
        agent=viewer,
    )

    # TASK 2: Teknik QA ve Final Karar
    tech_task = Task(
        description=(
            f"Teknik kalite kontrolü:\n"
            f"Başlık: {state.title}\n"
            f"Video: {state.output_path}\n"
            f"Sahne sayısı: {len(state.scenes)}\n\n"
            "Teknik kontroller:\n"
            "1. Ses seviyesi tutarlı mı?\n"
            "2. Geçişler pürüzsüz mü?\n"
            "3. Çözünürlük 1080p mi?\n"
            "4. Senkronizasyon hataları var mı?\n\n"
            "İzleyici değerlendirmesini de göz önünde bulundurarak "
            "SADECE şu JSON formatında yanıt ver:\n"
            "{\n"
            '  "overall_score": 7.5,\n'
            '  "approved": true,\n'
            '  "viewer_notes": "İzleyici yorumu özeti",\n'
            '  "technical_issues": ["sorun1"],\n'
            '  "revision_needed": false,\n'
            '  "revision_instructions": ""\n'
            "}"
        ),
        expected_output="JSON formatında QA raporu",
        agent=tech_qa,
        context=[viewer_task],
    )

    return Crew(
        agents=[viewer, tech_qa],
        tasks=[viewer_task, tech_task],
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
