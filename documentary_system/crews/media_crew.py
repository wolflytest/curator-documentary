"""
Medya ekibi: Searcher → Analyzer → ConsistencyAgent → Jury kararı.
Her sahne için ayrı çalışır.
"""
import logging

from crewai import Agent, Crew, Process, Task

from documentary_system.llm_config import get_llm
from documentary_system.state.documentary_state import DocumentaryState, SceneState
from documentary_system.tools.gemini_vision_tool import GeminiVisionTool
from documentary_system.tools.media_download_tool import MediaDownloadTool
from documentary_system.tools.pexels_tool import PexelsTool
from documentary_system.tools.wikimedia_tool import WikimediaTool
from documentary_system.tools.ytcc_tool import YouTubeCCTool

log = logging.getLogger(__name__)


def create_media_crew(
    scene: SceneState,
    state: DocumentaryState,
) -> Crew:
    """Bir sahne için medya seçim ekibini oluştur."""
    llm = get_llm()
    search_tools = [WikimediaTool(), PexelsTool(), YouTubeCCTool(), MediaDownloadTool()]
    vision_tools = [GeminiVisionTool(), MediaDownloadTool()]

    # AGENT 1: Arşiv Araştırmacısı
    searcher = Agent(
        role="Arşiv Araştırmacısı",
        goal=(
            f"'{scene.visual_description}' için en uygun tarihi "
            f"görsel/video materyalini bul ve indir. "
            f"Anahtar kelimeler: {', '.join(scene.search_keywords)}"
        ),
        backstory=(
            "Wikimedia, Getty ve tarihi arşivlerde 10 yıl çalışmış. "
            "Doğru keyword ile doğru materyali bulur. "
            "Telif sorunlarından kaçınır, sadece açık lisanslı içerik kullanır. "
            "Her aramada en az 5 aday çekmeden durmaz."
        ),
        llm=llm,
        tools=search_tools,
        verbose=True,
        max_iter=5,
    )

    # AGENT 2: Görsel İçerik Analisti
    analyzer = Agent(
        role="Görsel İçerik Analisti",
        goal=(
            "İndirilen her materyalin sahneyle alakasını, "
            "kalitesini ve kullanılabilirliğini değerlendir. "
            "10'lu skorlama sistemi kullan, gerekçesiz puan verme."
        ),
        backstory=(
            "Görsel sanatlar ve belgesel prodüksiyon uzmanı. "
            "Watermark, düşük kalite, alakasız içeriği saniyeler içinde fark eder. "
            "Sadece gerçekten kullanılabilir materyalleri önerir."
        ),
        llm=llm,
        tools=vision_tools,
        verbose=True,
        max_iter=3,
    )

    # AGENT 3: Görsel Tutarlılık Denetçisi
    consistency_agent = Agent(
        role="Görsel Tutarlılık Denetçisi",
        goal=(
            "Seçilecek görselin önceki sahnelerle stil, "
            "renk tonu ve dönem tutarlılığını kontrol et. "
            "Final JSON kararını ver."
        ),
        backstory=(
            "Film editörü. İzleyicinin fark etmediği ama hissettiği "
            "görsel tutarsızlıkları yakalar. Sepia serisine renkli fotoğraf, "
            "1800'lere ait sahnede modern bina geçmesine izin vermez.\n\n"
            f"Önceki sahnelerin görsel bağlamı:\n{state.get_visual_context_summary()}"
        ),
        llm=llm,
        tools=vision_tools,
        verbose=True,
        max_iter=3,
    )

    used_hashes = list(state.used_media_hashes)[:10]

    # TASK 1: Medya Arama + İndirme
    search_task = Task(
        description=(
            f"Sahne {scene.index} için medya ara ve indir:\n"
            f"Anlatım: {scene.narration[:150]}\n"
            f"Anahtar kelimeler: {scene.search_keywords}\n"
            f"Görsel açıklama: {scene.visual_description}\n"
            f"Duygu tonu: {scene.mood}\n\n"
            f"Kullanılmış medya (bunları seçme): {used_hashes}\n\n"
            "ADIMLAR:\n"
            "1. WikimediaTool ve PexelsTool ile arama yap, en az 5 aday bul\n"
            "2. YouTubeCCTool ile 1-2 video aday ara\n"
            "3. Her adayın download_url'ini MediaDownloadTool ile indir\n"
            "   - MediaDownloadTool input: {\"url\": \"<download_url>\", \"media_type\": \"photo|video\"}\n"
            "   - MediaDownloadTool local_path döndürür\n"
            "4. Her aday için: başlık, kaynak, download_url, local_path, tip bilgisini listele\n\n"
            "ÖNEMLİ: GeminiVisionTool local_path ister, URL değil. "
            "İndirmeden analiz yapılamaz."
        ),
        expected_output=(
            "En az 5 medya adayı listesi. Her adayda: "
            "başlık, kaynak (wikimedia/pexels/ytcc), download_url, local_path, media_type."
        ),
        agent=searcher,
    )

    # TASK 2: Görsel Analiz (local_path ile)
    analyze_task = Task(
        description=(
            f"Sahne {scene.index} için bulunan medya adaylarını analiz et:\n"
            f"Sahne metni: {scene.narration[:100]}\n"
            f"Aranan: {scene.visual_description}\n\n"
            "Her aday için GeminiVisionTool ile analiz yap:\n"
            "- GeminiVisionTool input: {\"image_path\": \"<local_path>\", "
            "\"scene_text\": \"...\", \"keywords\": [...]}\n"
            "- image_path: searcher'ın döndürdüğü local_path (NOT: URL değil!)\n"
            "- Eğer bir adayın local_path'i boşsa veya indirilemişse atla\n\n"
            "Skorlar:\n"
            "- Alakalılık skoru (1-10)\n"
            "- Kalite skoru (1-10)\n"
            "- Watermark var mı?\n"
            "- Kullanılabilir mi?\n\n"
            "Tüm adayları karşılaştır, en yüksek toplam skoru bul."
        ),
        expected_output="Her adayın analiz sonucu (local_path ile) ve en yüksek skorlu adayın belirlenmesi.",
        agent=analyzer,
        context=[search_task],
    )

    # TASK 3: Tutarlılık Kontrolü ve Final Karar
    consistency_task = Task(
        description=(
            f"Sahne {scene.index} için analyzer'ın seçtiği en iyi medyayı "
            f"önceki sahnelerle tutarlılık açısından kontrol et.\n"
            f"Seçilen medyanın local_path'i üzerinden GeminiVisionTool ile ek analiz yapabilirsin.\n\n"
            f"Önceki görsel stil:\n{state.get_visual_context_summary()}\n\n"
            "Kontrol et:\n"
            "- Renk tonu tutarlılığı\n"
            "- Dönem/tarih uyumu\n"
            "- Stil uyumu (karışık gravür/fotoğraf/modern sorun yaratır mı?)\n\n"
            "ÇIKTI — sadece JSON, başka hiçbir şey yazma:\n"
            "{\n"
            '  "selected_media": {\n'
            '    "local_path": "/tmp/...",\n'
            '    "source": "wikimedia|pexels|ytcc",\n'
            '    "source_url": "https://...",\n'
            '    "relevance_score": 8.5,\n'
            '    "quality_score": 7.0,\n'
            '    "consistency_score": 9.0,\n'
            '    "media_type": "photo|video",\n'
            '    "clip_start": 0.0,\n'
            '    "clip_end": 7.0,\n'
            '    "dominant_colors": ["brown", "gold"]\n'
            "  },\n"
            '  "rejected_count": 3,\n'
            '  "rejection_reasons": ["watermark", "düşük kalite"]\n'
            "}"
        ),
        expected_output="JSON formatında seçilmiş medya bilgisi",
        agent=consistency_agent,
        context=[search_task, analyze_task],
    )

    return Crew(
        agents=[searcher, analyzer, consistency_agent],
        tasks=[search_task, analyze_task, consistency_task],
        process=Process.sequential,
        verbose=True,
    )


if __name__ == "__main__":
    from documentary_system.state.documentary_state import DocumentaryState, SceneState

    scene = SceneState(
        index=0,
        narration="Osmanlı İmparatorluğu'nun kuruluşu...",
        search_keywords=["ottoman empire founding historical"],
        visual_description="Tarihi harita veya gravür",
        mood="dramatic",
        transition="fade",
        duration_sec=7.0,
    )
    state = DocumentaryState(doc_id=1, topic="Test")
    crew = create_media_crew(scene, state)
    print("✅ Media crew oluşturuldu")
    print(f"   Ajanlar: {[a.role for a in crew.agents]}")
