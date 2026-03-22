"""
SQLite veritabanı işlemleri.
"""
import sqlite3
import logging
from collections import Counter
from datetime import date
from pathlib import Path
from config import DB_PATH

log = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Tabloları oluştur, eksik sütunları ekle (migration)."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS contents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                url         TEXT NOT NULL,
                platform    TEXT NOT NULL,
                title       TEXT,
                transcript  TEXT,
                analysis    TEXT,
                priority    INTEGER,
                tags        TEXT,
                note        TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)
        # Eski veritabanlarına sütun ekle (varsa hata verme)
        for col, typedef in [("tags", "TEXT"), ("note", "TEXT")]:
            try:
                conn.execute(f"ALTER TABLE contents ADD COLUMN {col} {typedef}")
                log.info("Migration: '%s' sütunu eklendi.", col)
            except sqlite3.OperationalError:
                pass  # Sütun zaten var
        # Belgesel tabloları
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documentaries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                topic       TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                script_json TEXT,
                output_path TEXT,
                youtube_url TEXT,
                error_msg   TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                updated_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documentary_media (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                documentary_id  INTEGER NOT NULL,
                scene_index     INTEGER NOT NULL,
                source          TEXT NOT NULL,
                source_url      TEXT NOT NULL,
                local_path      TEXT,
                relevance_score REAL,
                used            INTEGER DEFAULT 0,
                created_at      TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id     INTEGER PRIMARY KEY,
                language    TEXT NOT NULL DEFAULT 'tr',
                voice       TEXT NOT NULL DEFAULT 'gtts_tr',
                duration    INTEGER NOT NULL DEFAULT 600,
                updated_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            )
        """)
        conn.commit()
    log.info("Veritabanı hazır: %s", DB_PATH)


def save_content(
    url: str,
    platform: str,
    title: str,
    transcript: str,
    analysis: str,
    priority: int,
    tags: list[str] | None = None,
    note: str | None = None,
) -> int:
    """Analiz edilmiş içeriği kaydet, yeni satırın id'sini döndür."""
    tags_str = ",".join(t.lower() for t in tags) if tags else ""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO contents
                (url, platform, title, transcript, analysis, priority, tags, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (url, platform, title, transcript, analysis, priority, tags_str, note or ""),
        )
        conn.commit()
        log.info("Kaydedildi → id=%s platform=%s tags=%s", cur.lastrowid, platform, tags_str)
        return cur.lastrowid


def get_all_contents(limit: int = 200) -> list[dict]:
    """Tüm kayıtları önceliğe göre sıralı getir (sohbet bağlamı için)."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM contents
            ORDER BY priority DESC, created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_daily_contents(for_date: date | None = None) -> list[dict]:
    """Belirtilen güne ait (varsayılan: bugün) tüm içerikleri getir."""
    target = for_date or date.today()
    date_str = target.strftime("%Y-%m-%d")
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM contents
            WHERE date(created_at) = ?
            ORDER BY priority DESC, created_at ASC
            """,
            (date_str,),
        ).fetchall()
    return [dict(row) for row in rows]


def search_by_tag(tag: str) -> list[dict]:
    """Belirli bir tag'e sahip kayıtları önceliğe göre sıralı getir."""
    tag = tag.lower().lstrip("#")
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM contents
            WHERE ',' || tags || ',' LIKE ?
            ORDER BY priority DESC, created_at DESC
            """,
            (f"%,{tag},%",),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_content(content_id: int) -> bool:
    """Kaydı sil. Silinen satır varsa True döndür."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM contents WHERE id = ?", (content_id,))
        conn.commit()
    deleted = cur.rowcount > 0
    if deleted:
        log.info("Kayıt silindi: id=%d", content_id)
    else:
        log.warning("Silinecek kayıt bulunamadı: id=%d", content_id)
    return deleted


def create_documentary(topic: str) -> int:
    """Yeni belgesel kaydı oluştur, id döndür."""
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO documentaries (topic, status) VALUES (?, 'pending')",
            (topic,),
        )
        conn.commit()
        log.info("Belgesel oluşturuldu: id=%d, konu='%s'", cur.lastrowid, topic)
        return cur.lastrowid


def update_documentary_status(
    doc_id: int,
    status: str,
    **kwargs: str,
) -> None:
    """Belgesel durumunu ve opsiyonel alanları güncelle."""
    allowed = {"script_json", "output_path", "youtube_url", "error_msg"}
    updates = {"status": status, "updated_at": "datetime('now','localtime')"}
    params = []
    set_parts = ["status = ?", "updated_at = datetime('now','localtime')"]
    params.append(status)

    for key, val in kwargs.items():
        if key in allowed and val is not None:
            set_parts.append(f"{key} = ?")
            params.append(val)

    params.append(doc_id)
    with get_connection() as conn:
        conn.execute(
            f"UPDATE documentaries SET {', '.join(set_parts)} WHERE id = ?",
            params,
        )
        conn.commit()
    log.info("Belgesel #%d güncellendi: %s", doc_id, status)


def save_documentary_media(
    documentary_id: int,
    scene_index: int,
    source: str,
    source_url: str,
    local_path: str = "",
    relevance_score: float = 0.0,
) -> int:
    """Belgesel için seçilen medyayı kaydet."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO documentary_media
                (documentary_id, scene_index, source, source_url, local_path, relevance_score, used)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            """,
            (documentary_id, scene_index, source, source_url, local_path, relevance_score),
        )
        conn.commit()
        return cur.lastrowid


def get_documentary(doc_id: int) -> dict | None:
    """Belgesel kaydını id ile getir."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM documentaries WHERE id = ?", (doc_id,)
        ).fetchone()
    return dict(row) if row else None


def list_documentaries(limit: int = 20) -> list[dict]:
    """Son belgeselleri listele."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM documentaries ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_user_settings(user_id: int) -> dict:
    """Kullanıcı ayarlarını getir (yoksa varsayılanları döndür)."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM user_settings WHERE user_id = ?", (user_id,)
        ).fetchone()
    if row:
        return dict(row)
    return {"user_id": user_id, "language": "tr", "voice": "gtts_tr", "duration": 600}


def save_user_settings(user_id: int, language: str, voice: str, duration: int) -> None:
    """Kullanıcı ayarlarını kaydet veya güncelle."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO user_settings (user_id, language, voice, duration, updated_at)
            VALUES (?, ?, ?, ?, datetime('now','localtime'))
            ON CONFLICT(user_id) DO UPDATE SET
                language   = excluded.language,
                voice      = excluded.voice,
                duration   = excluded.duration,
                updated_at = excluded.updated_at
            """,
            (user_id, language, voice, duration),
        )
        conn.commit()
    log.info("Kullanıcı #%d ayarları güncellendi: lang=%s voice=%s dur=%d", user_id, language, voice, duration)


def get_stats() -> dict:
    """İstatistik verisini döndür."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM contents").fetchone()[0]
        avg_priority = conn.execute("SELECT AVG(priority) FROM contents").fetchone()[0]
        platforms = conn.execute(
            "SELECT platform, COUNT(*) as cnt FROM contents GROUP BY platform ORDER BY cnt DESC"
        ).fetchall()
        tag_rows = conn.execute(
            "SELECT tags FROM contents WHERE tags IS NOT NULL AND tags != ''"
        ).fetchall()

    # Tag frekansı
    tag_counter: Counter = Counter()
    for row in tag_rows:
        for tag in row["tags"].split(","):
            tag = tag.strip()
            if tag:
                tag_counter[tag] += 1

    return {
        "total": total,
        "avg_priority": round(avg_priority, 1) if avg_priority else 0,
        "platforms": [(r["platform"], r["cnt"]) for r in platforms],
        "top_tags": tag_counter.most_common(10),
    }
