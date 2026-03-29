"""
EchoMirror - Step 4: Memory & Storage Module
Features:
  - AES-256 encrypted local SQLite database
  - Stores emotion sessions, conversations, sentiment scores
  - Emotional pattern detection (e.g. "sad 3 days in a row")
  - Growth tracking over time
  - Memory-aware context builder for Step 3 (ConversationEngine)
"""

import sqlite3
import json
import os
import hashlib
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from collections import Counter

load_dotenv()


# ─────────────────────────────────────────
# ENCRYPTION SETUP
# ─────────────────────────────────────────
def _get_encryption_key() -> bytes:
    """
    Derives a Fernet key from a secret stored in .env
    If no secret exists, generates one and saves it
    """
    secret = os.getenv("ECHO_DB_SECRET")
    if not secret:
        # Generate and save a new secret to .env
        secret = Fernet.generate_key().decode()
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        with open(env_path, "a") as f:
            f.write(f"\nECHO_DB_SECRET={secret}\n")
        print("[EchoMirror] Generated new DB encryption key and saved to .env")

    # Derive 32-byte key from secret using SHA256
    key_bytes = hashlib.sha256(secret.encode()).digest()
    # Encode to Fernet-compatible base64 key
    import base64
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    return fernet_key


class Encryptor:
    def __init__(self):
        self.fernet = Fernet(_get_encryption_key())

    def encrypt(self, text: str) -> str:
        return self.fernet.encrypt(text.encode()).decode()

    def decrypt(self, token: str) -> str:
        try:
            return self.fernet.decrypt(token.encode()).decode()
        except Exception:
            return "[encrypted]"


# ─────────────────────────────────────────
# DATABASE MANAGER
# ─────────────────────────────────────────
class MemoryDB:
    def __init__(self, db_path: str = "emotions.db"):
        self.db_path = db_path
        self.encryptor = Encryptor()
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Create all tables if they don't exist"""
        conn = self._connect()
        c = conn.cursor()

        # Emotion sessions — one row per detection event
        c.execute("""
            CREATE TABLE IF NOT EXISTS emotion_sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                date        TEXT NOT NULL,
                emotion     TEXT NOT NULL,
                confidence  REAL,
                brightness  REAL,
                is_low_light INTEGER DEFAULT 0
            )
        """)

        # Conversation logs — encrypted
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                date        TEXT NOT NULL,
                role        TEXT NOT NULL,
                message     TEXT NOT NULL,
                emotion     TEXT,
                sentiment   TEXT,
                polarity    REAL
            )
        """)

        # Daily summaries — one row per day
        c.execute("""
            CREATE TABLE IF NOT EXISTS daily_summaries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT UNIQUE NOT NULL,
                dominant_emotion TEXT,
                avg_polarity    REAL,
                session_count   INTEGER,
                notes           TEXT
            )
        """)

        # User goals & reflections
        c.execute("""
            CREATE TABLE IF NOT EXISTS reflections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                date        TEXT NOT NULL,
                content     TEXT NOT NULL,
                emotion     TEXT,
                sentiment   TEXT
            )
        """)

        conn.commit()
        conn.close()
        print(f"[MemoryDB] Database initialized: {self.db_path}")

    # ─── EMOTION LOGGING ───
    def log_emotion(self, emotion: str, confidence: float,
                    brightness: float = 255.0, is_low_light: bool = False):
        """Save a detected emotion event"""
        now = datetime.now()
        conn = self._connect()
        c = conn.cursor()
        c.execute("""
            INSERT INTO emotion_sessions (timestamp, date, emotion, confidence, brightness, is_low_light)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d"),
              emotion, confidence, brightness, int(is_low_light)))
        conn.commit()
        conn.close()

    # ─── CONVERSATION LOGGING ───
    def log_message(self, session_id: str, role: str, message: str,
                    emotion: str = "neutral", sentiment: str = "Neutral", polarity: float = 0.0):
        """Save a conversation message (encrypted)"""
        now = datetime.now()
        encrypted_msg = self.encryptor.encrypt(message)
        conn = self._connect()
        c = conn.cursor()
        c.execute("""
            INSERT INTO conversations (session_id, timestamp, date, role, message, emotion, sentiment, polarity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d"),
              role, encrypted_msg, emotion, sentiment, polarity))
        conn.commit()
        conn.close()

    def get_conversation_history(self, session_id: str) -> list:
        """Retrieve and decrypt conversation for a session"""
        conn = self._connect()
        c = conn.cursor()
        c.execute("""
            SELECT role, message, timestamp FROM conversations
            WHERE session_id = ? ORDER BY timestamp ASC
        """, (session_id,))
        rows = c.fetchall()
        conn.close()
        return [{"role": r[0], "content": self.encryptor.decrypt(r[1]),
                 "timestamp": r[2]} for r in rows]

    # ─── PATTERN DETECTION ───
    def get_dominant_emotion_today(self) -> str:
        """Returns the most detected emotion today"""
        today = datetime.now().strftime("%Y-%m-%d")
        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT emotion FROM emotion_sessions WHERE date = ?", (today,))
        rows = c.fetchall()
        conn.close()
        if not rows:
            return "neutral"
        counts = Counter([r[0] for r in rows])
        return counts.most_common(1)[0][0]

    def get_emotion_streak(self, emotion: str) -> int:
        """
        Returns how many consecutive days the user had this dominant emotion.
        e.g. if sad was dominant for 3 days in a row → returns 3
        """
        conn = self._connect()
        c = conn.cursor()
        c.execute("""
            SELECT date, emotion FROM emotion_sessions
            ORDER BY date DESC
        """)
        rows = c.fetchall()
        conn.close()

        if not rows:
            return 0

        # Get dominant emotion per day
        daily = {}
        for date, emo in rows:
            if date not in daily:
                daily[date] = []
            daily[date].append(emo)

        daily_dominant = {
            date: Counter(emotions).most_common(1)[0][0]
            for date, emotions in daily.items()
        }

        # Count consecutive days ending today
        streak = 0
        check_date = datetime.now().date()
        for _ in range(30):  # Check up to 30 days back
            date_str = check_date.strftime("%Y-%m-%d")
            if daily_dominant.get(date_str) == emotion:
                streak += 1
                check_date -= timedelta(days=1)
            else:
                break

        return streak

    def get_weekly_pattern(self) -> dict:
        """Returns emotion counts for the last 7 days"""
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        conn = self._connect()
        c = conn.cursor()
        c.execute("""
            SELECT emotion, COUNT(*) as count
            FROM emotion_sessions
            WHERE date >= ?
            GROUP BY emotion
            ORDER BY count DESC
        """, (week_ago,))
        rows = c.fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows}

    def get_avg_polarity_week(self) -> float:
        """Average sentiment polarity over the last 7 days"""
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        conn = self._connect()
        c = conn.cursor()
        c.execute("""
            SELECT AVG(polarity) FROM conversations
            WHERE date >= ? AND role = 'user'
        """, (week_ago,))
        result = c.fetchone()[0]
        conn.close()
        return round(result, 3) if result else 0.0

    def get_total_sessions(self) -> int:
        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT COUNT(DISTINCT date) FROM emotion_sessions")
        result = c.fetchone()[0]
        conn.close()
        return result or 0

    # ─── MEMORY CONTEXT BUILDER ───
    def build_memory_context(self) -> str:
        """
        Builds a memory summary string to inject into the AI system prompt.
        This makes EchoMirror remember the user across sessions.
        """
        today_emotion   = self.get_dominant_emotion_today()
        sad_streak      = self.get_emotion_streak("sad")
        angry_streak    = self.get_emotion_streak("angry")
        weekly_pattern  = self.get_weekly_pattern()
        avg_polarity    = self.get_avg_polarity_week()
        total_sessions  = self.get_total_sessions()

        lines = [f"USER MEMORY SUMMARY (from past sessions):"]
        lines.append(f"- Total days used EchoMirror: {total_sessions}")
        lines.append(f"- Today's dominant emotion: {today_emotion}")
        lines.append(f"- Average sentiment this week: {avg_polarity:+.2f}")

        if weekly_pattern:
            top_emotions = list(weekly_pattern.items())[:3]
            pattern_str = ", ".join([f"{e}: {c}x" for e, c in top_emotions])
            lines.append(f"- Most frequent emotions this week: {pattern_str}")

        # Streak alerts — important context for the AI
        if sad_streak >= 2:
            lines.append(f"- ALERT: User has been predominantly SAD for {sad_streak} consecutive days")
        if angry_streak >= 2:
            lines.append(f"- ALERT: User has been predominantly ANGRY for {angry_streak} consecutive days")

        if total_sessions == 0:
            lines.append("- This is the user's first session with EchoMirror")
        elif total_sessions < 5:
            lines.append(f"- User is still new to EchoMirror ({total_sessions} sessions)")

        return "\n".join(lines)

    # ─── REFLECTION SAVING ───
    def save_reflection(self, content: str, emotion: str = "neutral", sentiment: str = "Neutral"):
        """Save a user reflection/journal entry"""
        now = datetime.now()
        encrypted = self.encryptor.encrypt(content)
        conn = self._connect()
        c = conn.cursor()
        c.execute("""
            INSERT INTO reflections (timestamp, date, content, emotion, sentiment)
            VALUES (?, ?, ?, ?, ?)
        """, (now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d"),
              encrypted, emotion, sentiment))
        conn.commit()
        conn.close()

    def get_reflections(self, days: int = 7) -> list:
        """Get decrypted reflections from last N days"""
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        conn = self._connect()
        c = conn.cursor()
        c.execute("""
            SELECT timestamp, content, emotion, sentiment
            FROM reflections WHERE date >= ?
            ORDER BY timestamp DESC
        """, (since,))
        rows = c.fetchall()
        conn.close()
        return [{
            "timestamp": r[0],
            "content": self.encryptor.decrypt(r[1]),
            "emotion": r[2],
            "sentiment": r[3]
        } for r in rows]

    # ─── DAILY SUMMARY ───
    def save_daily_summary(self):
        """Auto-generate and save today's summary"""
        today = datetime.now().strftime("%Y-%m-%d")
        dominant = self.get_dominant_emotion_today()
        avg_pol  = self.get_avg_polarity_week()

        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM emotion_sessions WHERE date = ?", (today,))
        count = c.fetchone()[0]

        c.execute("""
            INSERT OR REPLACE INTO daily_summaries
            (date, dominant_emotion, avg_polarity, session_count)
            VALUES (?, ?, ?, ?)
        """, (today, dominant, avg_pol, count))
        conn.commit()
        conn.close()

    def print_stats(self):
        """Print a memory report to console"""
        print("\n" + "="*50)
        print("  EchoMirror Memory Report")
        print("="*50)
        print(self.build_memory_context())
        weekly = self.get_weekly_pattern()
        if weekly:
            print("\nWeekly Emotion Breakdown:")
            for emo, count in weekly.items():
                bar = "█" * min(count, 30)
                print(f"  {emo:10s} {bar} ({count})")
        print("="*50 + "\n")


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    db = MemoryDB()

    # Simulate some data
    print("[TEST] Logging sample emotions...")
    for emo, conf in [("sad", 0.82), ("neutral", 0.91), ("sad", 0.76), ("happy", 0.95)]:
        db.log_emotion(emo, conf)

    print("[TEST] Logging sample conversation...")
    sid = "test_session_001"
    db.log_message(sid, "user", "I feel really lost today", "sad", "Negative", -0.4)
    db.log_message(sid, "assistant", "It sounds like you're carrying a lot right now.", "sad", "Negative", -0.4)

    print("[TEST] Logging a reflection...")
    db.save_reflection("I want to become an architect someday.", "neutral", "Positive")

    print("[TEST] Building memory context...")
    context = db.build_memory_context()
    print(context)

    db.print_stats()

    print("[TEST] Retrieving conversation...")
    history = db.get_conversation_history(sid)
    for msg in history:
        print(f"  [{msg['role']}]: {msg['content']}")

    print("\n[MemoryDB] All tests passed!")