"""
LLM yapılandırması: Ollama (yerel) + Gemini 2.0 Flash (fallback).
Önce Ollama'ya bağlanmayı dener, başarısız olursa Gemini kullanır.
"""
import logging
import os

import httpx
from crewai import LLM

log = logging.getLogger(__name__)


def get_local_llm() -> LLM:
    """Ollama üzerinden Qwen2.5:14b yerel modeli döndür."""
    return LLM(
        model="ollama/qwen2.5:14b",
        base_url="http://localhost:11434",
        temperature=0.7,
    )


def get_fallback_llm() -> LLM:
    """Gemini 2.0 Flash — Ollama çalışmazsa kullanılır."""
    return LLM(
        model="gemini/gemini-2.0-flash",
        api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0.7,
    )


def get_llm() -> LLM:
    """
    Önce Ollama bağlantısını kontrol et, qwen2.5:14b varsa kullan.
    Ollama çalışmıyor veya model yoksa Gemini 2.0 Flash'a geç.
    """
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = r.json().get("models", [])
            names = [m.get("name", "") for m in models]
            if any("qwen2.5" in n for n in names):
                log.info("Ollama: qwen2.5:14b bulundu, yerel LLM kullanılıyor.")
                return get_local_llm()
            log.warning("Ollama çalışıyor ama qwen2.5:14b yok. Mevcut: %s", names)
    except Exception as exc:
        log.warning("Ollama bağlanamadı (%s) → Gemini fallback.", exc)
    return get_fallback_llm()


if __name__ == "__main__":
    llm = get_llm()
    print(f"✅ LLM: {llm.model}")
