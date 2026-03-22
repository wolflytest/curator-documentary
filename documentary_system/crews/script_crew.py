"""
Senaryo ekibi: Creator → Optimizer → Critic (max 3 tur revizyon).
Her agent ayrı task olarak tanımlanır, sequential process ile çalışır.
"""
import logging

from crewai import Agent, Crew, Process, Task

from documentary_system.llm_config import get_llm

log = logging.getLogger(__name__)


def create_script_crew(
    topic: str,
    target_duration: int = 600,
    language: str = "tr",
) -> tuple[Crew, Task]:
    """
    Senaryo üretim ekibini oluştur.
    Returns: (crew, critic_task) — crew.kickoff() sonrası critic_task.output okunur.
    """
    llm = get_llm()
    scene_count = max(5, target_duration // 7)
    lang_label = "Türkçe" if language == "tr" else "English"

    # AGENT 1: Senarist
    creator = Agent(
        role="Tarihi Belgesel Senarist",
        goal=(
            f"'{topic}' konusunda izleyiciyi ilk 30 saniyede yakalayan, "
            f"BBC/NatGeo kalitesinde {lang_label} belgesel senaryosu yaz. "
            f"Yaklaşık {scene_count} sahne, toplam {target_duration} saniye."
        ),
        backstory=(
            "15 yıllık belgesel yapım deneyimine sahip, Ken Burns tarzı "
            "anlatım ustası. Her sahnede görsel-ses uyumunu düşünürsün. "
            "Hook olmadan senaryo yazmaz, her cümle izleyiciyi bağlar. "
            "Tarihi doğruluğa önem verirsin."
        ),
        llm=llm,
        verbose=True,
        max_iter=3,
    )

    # AGENT 2: SEO ve İzlenme Uzmanı
    optimizer = Agent(
        role="YouTube SEO ve İzlenme Uzmanı",
        goal=(
            "Senaristin yazdığı senaryonun YouTube algoritmasına uygunluğunu "
            "artır, izlenme süresi ve CTR'ı optimize et. "
            "Hook'ları güçlendir, başlık ve açıklamayı SEO için optimize et."
        ),
        backstory=(
            "YouTube algoritması uzmanı. Hangi hook'ların izleyiciyi tuttuğunu, "
            "hangi başlıkların tıklandığını verilere dayalı bilirsin. "
            "İlk 30 saniye retention takıntısısın. "
            "SEO için doğru keyword'leri bilirsin."
        ),
        llm=llm,
        verbose=True,
        max_iter=3,
    )

    # AGENT 3: Baş Editör ve Eleştirmen
    critic = Agent(
        role="Baş Editör ve Kalite Denetçisi",
        goal=(
            "Senaryonun tarihi doğruluğunu, anlatım akışını ve görsel uyum "
            "potansiyelini denetle. Son JSON çıktısını üret."
        ),
        backstory=(
            "20 yıllık editör. Zayıf senaryoları ilk cümlede fark eder. "
            "Tarihi hata affetmez. Ama iyi işi de teslim etmesini bilir. "
            "Çıktıyı her zaman tam ve geçerli JSON olarak verir."
        ),
        llm=llm,
        verbose=True,
        max_iter=3,
    )

    # TASK 1: Senaryo Yazımı
    creator_task = Task(
        description=(
            f"Konu: {topic}\n"
            f"Hedef süre: {target_duration} saniye ({scene_count} sahne)\n\n"
            f"BBC/NatGeo kalitesinde {lang_label} belgesel senaryosu yaz.\n"
            "Her sahne için şunları belirt:\n"
            f"- {lang_label} anlatım metni (narration)\n"
            "- İngilizce medya arama anahtar kelimeleri (search_keywords)\n"
            "- Görsel açıklama: ekranda ne gösterilmeli (visual_description)\n"
            "- Duygu tonu: dramatic|peaceful|tense|neutral (mood)\n"
            "- Geçiş türü: fade|cut|dissolve (transition)\n"
            "- Tahmini süre saniye olarak (duration_sec)\n\n"
            "İlk sahne güçlü bir hook ile başlamalı. "
            "Son sahne güçlü bir kapanışla bitmeli."
        ),
        expected_output=(
            f"{scene_count}-scene {lang_label} documentary script. "
            "Each scene: narration, search_keywords, visual_description, mood, transition, duration_sec."
        ),
        agent=creator,
    )

    # TASK 2: SEO Optimizasyonu
    optimizer_task = Task(
        description=(
            f"Konu: {topic}\n\n"
            "Senaristin yazdığı senaryoyu al ve şunları optimize et:\n"
            "1. YouTube başlığı — 60 karakter max, dikkat çekici, keyword içermeli\n"
            "2. YouTube açıklaması — 500 karakter, SEO için optimize\n"
            "3. YouTube tag'leri — en az 15 tag\n"
            "4. İlk sahnenin hook'unu güçlendir\n"
            "5. Sahne sürelerini izleyici davranışına göre optimize et\n\n"
            "Senaryonun tarihi içeriğini değiştirme, sadece SEO ve anlatımı güçlendir."
        ),
        expected_output=(
            "SEO optimize edilmiş başlık, açıklama, tag listesi ve "
            "güçlendirilmiş hook ile güncellenmiş sahne listesi."
        ),
        agent=optimizer,
        context=[creator_task],
    )

    # TASK 3: Kalite Kontrolü ve Final JSON
    critic_task = Task(
        description=(
            f"Konu: {topic}\n\n"
            "Senarist ve optimizer'ın çalışmalarını değerlendir. "
            "Tarihi doğruluk, anlatım akışı ve görsel uyumu kontrol et.\n\n"
            "ÇIKTI FORMAT — sadece geçerli JSON, başka hiçbir şey yazma:\n"
            "{\n"
            '  "title": "YouTube başlığı (60 karakter max)",\n'
            '  "description": "YouTube açıklaması (500 karakter)",\n'
            '  "tags": ["tag1", "tag2"],\n'
            '  "revision_count": 0,\n'
            '  "approved": true,\n'
            '  "critic_notes": "",\n'
            '  "scenes": [\n'
            '    {\n'
            '      "index": 0,\n'
            f'      "narration": "{lang_label} narration text",\n'
            '      "search_keywords": ["english", "keyword"],\n'
            '      "visual_description": "Görselde ne olmalı",\n'
            '      "mood": "dramatic",\n'
            '      "transition": "fade",\n'
            '      "duration_sec": 7.0\n'
            '    }\n'
            '  ]\n'
            "}"
        ),
        expected_output="Geçerli JSON formatında belgesel senaryosu",
        agent=critic,
        context=[creator_task, optimizer_task],
    )

    crew = Crew(
        agents=[creator, optimizer, critic],
        tasks=[creator_task, optimizer_task, critic_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew, critic_task


if __name__ == "__main__":
    crew, task = create_script_crew("Osmanlı İmparatorluğu'nun Yükselişi", target_duration=60)
    print("✅ Script crew oluşturuldu")
    print(f"   Ajanlar: {[a.role for a in crew.agents]}")
    print(f"   Görevler: {len(crew.tasks)}")
