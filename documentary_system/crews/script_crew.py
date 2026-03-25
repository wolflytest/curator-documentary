"""
Senaryo ekibi: Creator → Optimizer → Critic (sequential).
Dil parametresine göre tüm agent/task içerikleri target language'da oluşturulur.
"""
import logging

from crewai import Agent, Crew, Process, Task

from documentary_system.llm_config import get_llm

log = logging.getLogger(__name__)

# ── Dil sabitleri ─────────────────────────────────────────────────────────────
_LANG = {
    "tr": {
        "narration_lang":     "Türkçe",
        "creator_role":       "Tarihi Belgesel Senarist",
        "creator_goal":       (
            "'{topic}' konusunda izleyiciyi ilk 30 saniyede yakalayan, "
            "BBC/NatGeo kalitesinde Türkçe belgesel senaryosu yaz. "
            "Yaklaşık {scene_count} sahne, toplam {target_duration} saniye."
        ),
        "creator_backstory":  (
            "15 yıllık belgesel yapım deneyimine sahip, Ken Burns tarzı "
            "anlatım ustası. Her sahnede görsel-ses uyumunu düşünür. "
            "Hook olmadan senaryo yazmaz. Tarihi doğruluğa önem verir."
        ),
        "optimizer_role":     "YouTube SEO ve İzlenme Uzmanı",
        "optimizer_goal":     (
            "Senaristin yazdığı Türkçe senaryonun YouTube algoritmasına "
            "uygunluğunu artır, hook'ları güçlendir, başlık/açıklamayı SEO için optimize et."
        ),
        "optimizer_backstory": (
            "YouTube algoritması uzmanı. İlk 30 saniye retention takıntısı. "
            "SEO için doğru keyword'leri ve thumbnail hook'larını bilir."
        ),
        "critic_role":        "Baş Editör ve Kalite Denetçisi",
        "critic_goal":        (
            "Senaryonun tarihi doğruluğunu, anlatım akışını ve görsel uyum "
            "potansiyelini denetle. Son JSON çıktısını üret."
        ),
        "critic_backstory":   (
            "20 yıllık editör. Tarihi hata affetmez. "
            "BBC World Service kalite standartlarını uygular. "
            "Çıktı her zaman tam ve geçerli JSON. "
            "Zayıf bir sahne için 'bu kabul edilemez' der ve yeniden yazılmasını ister. "
            "Minimum kalite standardı: her sahne görsel-ses uyumlu, her narration canlı ve özgün olmalı."
        ),
        "creator_task_desc":  (
            "Konu: {topic}\n"
            "Hedef süre: {target_duration} saniye ({scene_count} sahne)\n\n"
            "BBC/NatGeo kalitesinde Türkçe belgesel senaryosu yaz.\n"
            "Her sahne için şunları belirt:\n"
            "- Türkçe anlatım metni (narration)\n"
            "- İngilizce medya arama anahtar kelimeleri (search_keywords — İngilizce olmalı)\n"
            "- Görsel açıklama: ekranda ne gösterilmeli (visual_description)\n"
            "- Duygu tonu: dramatic|peaceful|tense|neutral (mood)\n"
            "- Geçiş türü: fade|cut|dissolve (transition)\n"
            "- Tahmini süre saniye olarak (duration_sec)\n\n"
            "İlk sahne güçlü hook, son sahne güçlü kapanış."
        ),
        "optimizer_task_desc": (
            "Konu: {topic}\n\n"
            "Türkçe senaryoyu al ve şunları optimize et:\n"
            "1. YouTube başlığı — 60 karakter max, dikkat çekici, keyword içermeli\n"
            "2. YouTube açıklaması — 500 karakter, SEO optimize\n"
            "3. YouTube tag'leri — en az 15 tag\n"
            "4. İlk sahnenin hook'unu güçlendir\n"
            "5. Sahne sürelerini izleyici davranışına göre optimize et"
        ),
        "narration_example":  "Türkçe anlatım metni buraya",
        "title_example":      "YouTube başlığı (60 karakter max)",
        "desc_example":       "YouTube açıklaması (500 karakter)",
        "visual_example":     "Görselde ne olmalı",
    },
    "en": {
        "narration_lang":     "English",
        "creator_role":       "Historical Documentary Scriptwriter",
        "creator_goal":       (
            "Write a BBC/NatGeo-quality English documentary script about '{topic}' "
            "that hooks the viewer in the first 30 seconds. "
            "Approximately {scene_count} scenes, total {target_duration} seconds."
        ),
        "creator_backstory":  (
            "15 years of documentary production experience, Ken Burns storytelling style. "
            "Thinks about visual-audio harmony in every scene. "
            "Never writes without a hook. Committed to historical accuracy."
        ),
        "optimizer_role":     "YouTube SEO and Audience Retention Expert",
        "optimizer_goal":     (
            "Optimize the English documentary script for YouTube's algorithm. "
            "Strengthen hooks, optimize title and description for SEO and CTR."
        ),
        "optimizer_backstory": (
            "YouTube algorithm specialist obsessed with first-30-second retention. "
            "Knows which hooks keep viewers watching and which titles get clicks."
        ),
        "critic_role":        "Senior Editor and Quality Controller",
        "critic_goal":        (
            "Review the script for historical accuracy, narrative flow, and visual "
            "alignment potential. Produce the final JSON output."
        ),
        "critic_backstory":   (
            "20-year veteran senior editor at BBC and National Geographic. "
            "Does not forgive historical inaccuracies or weak narration. "
            "Enforces BBC World Service quality standards strictly. "
            "Always outputs complete, valid JSON and nothing else. "
            "Rewrites weak scenes without hesitation. "
            "Minimum standard: every scene must have visual-audio harmony and vivid original narration."
        ),
        "creator_task_desc":  (
            "Topic: {topic}\n"
            "Target duration: {target_duration} seconds ({scene_count} scenes)\n\n"
            "CRITICAL REQUIREMENT: Write ALL narration exclusively in ENGLISH.\n"
            "Even if the topic name is in another language, every narration field MUST be in English.\n\n"
            "Write a BBC/NatGeo-quality English documentary script.\n"
            "For each scene specify:\n"
            "- narration: English narration text ONLY (never write in Turkish or any other language)\n"
            "- search_keywords: English media search keywords\n"
            "- visual_description: what should be shown on screen (English)\n"
            "- mood: dramatic|peaceful|tense|neutral\n"
            "- transition: fade|cut|dissolve\n"
            "- duration_sec: estimated duration in seconds\n\n"
            "First scene: strong cinematic hook. Last scene: powerful closing."
        ),
        "optimizer_task_desc": (
            "Topic: {topic}\n\n"
            "CRITICAL: ALL text must remain in English. Do NOT translate or rewrite in any other language.\n\n"
            "Take the English script and optimize:\n"
            "1. YouTube title — max 60 characters, attention-grabbing, keyword-rich (English)\n"
            "2. YouTube description — 500 characters, SEO optimized (English)\n"
            "3. YouTube tags — at least 15 tags (English)\n"
            "4. Strengthen the first scene's hook — make it cinematic and gripping\n"
            "5. Optimize scene durations based on viewer behavior patterns"
        ),
        "narration_example":  "English narration text here",
        "title_example":      "YouTube title (max 60 characters)",
        "desc_example":       "YouTube description (500 characters)",
        "visual_example":     "What should be shown on screen",
    },
}


def create_script_crew(
    topic: str,
    target_duration: int = 600,
    language: str = "tr",
) -> tuple[Crew, Task]:
    """
    Senaryo üretim ekibini oluştur.
    language="tr" → Türkçe senaryo, language="en" → English script
    Returns: (crew, critic_task)
    """
    llm = get_llm()
    scene_count = max(5, target_duration // 7)
    L = _LANG.get(language, _LANG["tr"])

    def _fmt(tmpl: str) -> str:
        return tmpl.format(topic=topic, scene_count=scene_count, target_duration=target_duration)

    # ── Agent 1: Senarist / Scriptwriter ────────────────────────────────────
    creator = Agent(
        role=L["creator_role"],
        goal=_fmt(L["creator_goal"]),
        backstory=L["creator_backstory"],
        llm=llm,
        verbose=True,
        max_iter=4,
    )

    # ── Agent 2: SEO Uzmanı ──────────────────────────────────────────────────
    optimizer = Agent(
        role=L["optimizer_role"],
        goal=_fmt(L["optimizer_goal"]),
        backstory=L["optimizer_backstory"],
        llm=llm,
        verbose=True,
        max_iter=3,
    )

    # ── Agent 3: Editör / Editor ─────────────────────────────────────────────
    critic = Agent(
        role=L["critic_role"],
        goal=_fmt(L["critic_goal"]),
        backstory=L["critic_backstory"],
        llm=llm,
        verbose=True,
        max_iter=3,
    )

    # ── Task 1: Senaryo yazımı ───────────────────────────────────────────────
    creator_task = Task(
        description=_fmt(L["creator_task_desc"]),
        expected_output=(
            f"{scene_count}-scene {L['narration_lang']} documentary script with "
            "narration, search_keywords, visual_description, mood, transition, duration_sec per scene."
        ),
        agent=creator,
    )

    # ── Task 2: SEO optimizasyonu ────────────────────────────────────────────
    optimizer_task = Task(
        description=_fmt(L["optimizer_task_desc"]),
        expected_output=(
            f"SEO-optimized title, description, tags list and updated scene list "
            f"with stronger hook (all in {L['narration_lang']})."
        ),
        agent=optimizer,
        context=[creator_task],
    )

    # ── Task 3: Kalite kontrolü + Final JSON ─────────────────────────────────
    _lang_enforcement = (
        "CRITICAL LANGUAGE RULE: Every 'narration' field MUST be written in "
        f"{L['narration_lang']} ONLY. "
        "If any narration is not in the correct language, rewrite it before outputting JSON.\n\n"
    ) if language == "en" else (
        "KRİTİK DİL KURALI: Her 'narration' alanı YALNIZCA Türkçe olmalıdır.\n\n"
    )
    critic_task = Task(
        description=(
            f"Topic: {topic}\n\n"
            + _lang_enforcement +
            "Review the scriptwriter's and optimizer's work. "
            "Check historical accuracy, narrative flow, and visual alignment.\n\n"
            "OUTPUT FORMAT — valid JSON only, nothing else:\n"
            "{\n"
            f'  "title": "{L["title_example"]}",\n'
            f'  "description": "{L["desc_example"]}",\n'
            '  "tags": ["tag1", "tag2"],\n'
            '  "revision_count": 0,\n'
            '  "approved": true,\n'
            '  "critic_notes": "",\n'
            '  "scenes": [\n'
            '    {\n'
            '      "index": 0,\n'
            f'      "narration": "{L["narration_example"]}",\n'
            '      "search_keywords": ["english", "keyword"],\n'
            f'      "visual_description": "{L["visual_example"]}",\n'
            '      "mood": "dramatic",\n'
            '      "transition": "fade",\n'
            '      "duration_sec": 7.0\n'
            '    }\n'
            '  ]\n'
            "}"
        ),
        expected_output="Valid JSON documentary script",
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
    crew_tr, _ = create_script_crew("Osmanlı İmparatorluğu'nun Yükselişi", target_duration=60, language="tr")
    crew_en, _ = create_script_crew("Rise of the Ottoman Empire", target_duration=60, language="en")
    print("✅ TR crew:", [a.role for a in crew_tr.agents])
    print("✅ EN crew:", [a.role for a in crew_en.agents])
