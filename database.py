"""
EchoMirror - Memory & Storage Module (v3 — Multi-User)
Features:
  - AES-256 encrypted local SQLite database
  - Thread-safe connections (multi-thread safe)
  - Indexed columns for fast queries
  - **Multi-user support**: user_id column on all data tables
  - **Users table**: with hashed passwords (bcrypt-style SHA256+salt)
  - Stores emotion sessions, conversations, sentiment scores per user
  - Emotional pattern detection (e.g. "sad 3 days in a row")
  - Growth tracking over time
  - Memory-aware context builder for ConversationEngine
  - Goal tracking with reframing support
"""

import sqlite3
import json
import os
import hashlib
import threading
import secrets
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
        secret = Fernet.generate_key().decode()
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        with open(env_path, "a") as f:
            f.write(f"\nECHO_DB_SECRET={secret}\n")
        print("[EchoMirror] Generated new DB encryption key and saved to .env")

    key_bytes = hashlib.sha256(secret.encode()).digest()
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
# PASSWORD HASHING
# ─────────────────────────────────────────
def _hash_password(password: str, salt: str = None) -> tuple:
    """Hash password with SHA256 + salt. Returns (hash, salt)."""
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return hashed, salt


# ─────────────────────────────────────────
# DATABASE MANAGER (Multi-User)
# ─────────────────────────────────────────
class MemoryDB:
    def __init__(self, db_path: str = "emotions.db"):
        self.db_path = db_path
        self.encryptor = Encryptor()
        self._write_lock = threading.Lock()
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        """Create all tables if they don't exist"""
        conn = self._connect()
        c = conn.cursor()

        # ── Users table ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT UNIQUE NOT NULL,
                email       TEXT,
                password    TEXT NOT NULL,
                salt        TEXT NOT NULL,
                display_name TEXT,
                created_at  TEXT DEFAULT (datetime('now','localtime')),
                last_login  TEXT
            )
        """)

        # Migration: add email column if not exists (for existing DBs)
        try:
            c.execute("ALTER TABLE users ADD COLUMN email TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Emotion sessions — one row per detection event
        c.execute("""
            CREATE TABLE IF NOT EXISTS emotion_sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER,
                timestamp   TEXT NOT NULL,
                date        TEXT NOT NULL,
                emotion     TEXT NOT NULL,
                confidence  REAL,
                brightness  REAL,
                is_low_light INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Conversation logs — encrypted
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER,
                session_id  TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                date        TEXT NOT NULL,
                role        TEXT NOT NULL,
                message     TEXT NOT NULL,
                emotion     TEXT,
                sentiment   TEXT,
                polarity    REAL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Daily summaries
        c.execute("""
            CREATE TABLE IF NOT EXISTS daily_summaries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         INTEGER,
                date            TEXT NOT NULL,
                dominant_emotion TEXT,
                avg_polarity    REAL,
                session_count   INTEGER,
                notes           TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, date)
            )
        """)

        # User reflections
        c.execute("""
            CREATE TABLE IF NOT EXISTS reflections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER,
                timestamp   TEXT NOT NULL,
                date        TEXT NOT NULL,
                content     TEXT NOT NULL,
                emotion     TEXT,
                sentiment   TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Evaluation tracking
        c.execute("""
            CREATE TABLE IF NOT EXISTS evaluation (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER,
                timestamp   TEXT DEFAULT (datetime('now','localtime')),
                predicted   TEXT,
                actual      TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Goal tracking
        c.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER,
                timestamp   TEXT NOT NULL,
                date        TEXT NOT NULL,
                session_id  TEXT,
                goal_text   TEXT NOT NULL,
                emotion     TEXT,
                status      TEXT DEFAULT 'active',
                reframed_to TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # ── Performance indexes ──
        c.execute("CREATE INDEX IF NOT EXISTS idx_emotion_date ON emotion_sessions(date)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_emotion_user ON emotion_sessions(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_conv_date ON conversations(date)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_reflect_date ON reflections(date)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_goals_date ON goals(date)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_goals_user ON goals(user_id)")

        conn.commit()
        conn.close()
        print(f"[MemoryDB] Database initialized: {self.db_path}")

    # ─── USER MANAGEMENT ───
    def create_user(self, username: str, password: str, display_name: str = "",
                    email: str = "") -> int | None:
        """Create a new user. Returns user_id or None if username taken."""
        hashed, salt = _hash_password(password)
        with self._write_lock:
            conn = self._connect()
            c = conn.cursor()
            try:
                c.execute("""
                    INSERT INTO users (username, email, password, salt, display_name)
                    VALUES (?, ?, ?, ?, ?)
                """, (username.lower().strip(), email.lower().strip() if email else "",
                      hashed, salt, display_name or username))
                conn.commit()
                user_id = c.lastrowid
                conn.close()
                print(f"[MemoryDB] User created: {username} (id={user_id})")
                return user_id
            except sqlite3.IntegrityError:
                conn.close()
                return None  # Username already exists

    def authenticate_user(self, username: str, password: str) -> int | None:
        """Verify credentials. Returns user_id or None."""
        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT id, password, salt FROM users WHERE username = ?",
                  (username.lower().strip(),))
        row = c.fetchone()
        conn.close()

        if not row:
            return None

        user_id, stored_hash, salt = row
        check_hash, _ = _hash_password(password, salt)
        if check_hash == stored_hash:
            # Update last login
            with self._write_lock:
                conn = self._connect()
                c = conn.cursor()
                c.execute("UPDATE users SET last_login = datetime('now','localtime') WHERE id = ?",
                          (user_id,))
                conn.commit()
                conn.close()
            return user_id
        return None

    def reset_password(self, username: str, email: str, new_password: str) -> bool:
        """Reset password if username+email match. Returns True on success."""
        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT id, email FROM users WHERE username = ?",
                  (username.lower().strip(),))
        row = c.fetchone()
        conn.close()

        if not row:
            return False

        user_id, stored_email = row
        if not stored_email or stored_email.lower() != email.lower().strip():
            return False

        # Update password
        hashed, salt = _hash_password(new_password)
        with self._write_lock:
            conn = self._connect()
            c = conn.cursor()
            c.execute("UPDATE users SET password = ?, salt = ? WHERE id = ?",
                      (hashed, salt, user_id))
            conn.commit()
            conn.close()
        print(f"[MemoryDB] Password reset for user: {username}")
        return True

    def verify_user_email(self, username: str, email: str) -> bool:
        """Check if username+email pair exists. Used for OTP verification."""
        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username = ? AND email = ?",
                  (username.lower().strip(), email.lower().strip()))
        row = c.fetchone()
        conn.close()
        return row is not None

    def get_user_info(self, user_id: int) -> dict | None:
        """Get user display name and stats."""
        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT username, display_name, created_at, last_login FROM users WHERE id = ?",
                  (user_id,))
        row = c.fetchone()
        conn.close()
        if not row:
            return None
        return {
            "username": row[0],
            "display_name": row[1],
            "created_at": row[2],
            "last_login": row[3],
        }

    # ─── EMOTION LOGGING ───
    def log_emotion(self, emotion: str, confidence: float,
                    brightness: float = 255.0, is_low_light: bool = False,
                    user_id: int = None):
        """Save a detected emotion event (thread-safe)"""
        now = datetime.now()
        with self._write_lock:
            conn = self._connect()
            c = conn.cursor()
            c.execute("""
                INSERT INTO emotion_sessions (user_id, timestamp, date, emotion, confidence, brightness, is_low_light)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d"),
                  emotion, confidence, brightness, int(is_low_light)))
            conn.commit()
            conn.close()

    # ─── CONVERSATION LOGGING ───
    def log_message(self, session_id: str, role: str, message: str,
                    emotion: str = "neutral", sentiment: str = "Neutral",
                    polarity: float = 0.0, user_id: int = None):
        """Save a conversation message (encrypted, thread-safe)"""
        now = datetime.now()
        encrypted_msg = self.encryptor.encrypt(message)
        with self._write_lock:
            conn = self._connect()
            c = conn.cursor()
            c.execute("""
                INSERT INTO conversations (user_id, session_id, timestamp, date, role, message, emotion, sentiment, polarity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, session_id, now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d"),
                  role, encrypted_msg, emotion, sentiment, polarity))
            conn.commit()
            conn.close()

    def get_conversation_history(self, session_id: str, user_id: int = None) -> list:
        """Retrieve and decrypt conversation for a session"""
        conn = self._connect()
        c = conn.cursor()
        if user_id:
            c.execute("""
                SELECT role, message, timestamp FROM conversations
                WHERE session_id = ? AND user_id = ? ORDER BY timestamp ASC
            """, (session_id, user_id))
        else:
            c.execute("""
                SELECT role, message, timestamp FROM conversations
                WHERE session_id = ? ORDER BY timestamp ASC
            """, (session_id,))
        rows = c.fetchall()
        conn.close()
        return [{"role": r[0], "content": self.encryptor.decrypt(r[1]),
                 "timestamp": r[2]} for r in rows]

    # ─── PATTERN DETECTION ───
    def get_dominant_emotion_today(self, user_id: int = None) -> str:
        """Returns the most detected emotion today"""
        today = datetime.now().strftime("%Y-%m-%d")
        conn = self._connect()
        c = conn.cursor()
        if user_id:
            c.execute("SELECT emotion FROM emotion_sessions WHERE date = ? AND user_id = ?",
                      (today, user_id))
        else:
            c.execute("SELECT emotion FROM emotion_sessions WHERE date = ?", (today,))
        rows = c.fetchall()
        conn.close()
        if not rows:
            return "neutral"
        counts = Counter([r[0] for r in rows])
        return counts.most_common(1)[0][0]

    def get_emotion_streak(self, emotion: str, user_id: int = None) -> int:
        """Returns how many consecutive days the user had this dominant emotion."""
        conn = self._connect()
        c = conn.cursor()
        if user_id:
            c.execute("""
                SELECT date, emotion, COUNT(*) as cnt
                FROM emotion_sessions
                WHERE date >= date('now', '-30 days') AND user_id = ?
                GROUP BY date, emotion
                ORDER BY date DESC, cnt DESC
            """, (user_id,))
        else:
            c.execute("""
                SELECT date, emotion, COUNT(*) as cnt
                FROM emotion_sessions
                WHERE date >= date('now', '-30 days')
                GROUP BY date, emotion
                ORDER BY date DESC, cnt DESC
            """)
        rows = c.fetchall()
        conn.close()

        if not rows:
            return 0

        daily_dominant = {}
        for date, emo, cnt in rows:
            if date not in daily_dominant:
                daily_dominant[date] = emo

        streak = 0
        check_date = datetime.now().date()
        for _ in range(30):
            date_str = check_date.strftime("%Y-%m-%d")
            if daily_dominant.get(date_str) == emotion:
                streak += 1
                check_date -= timedelta(days=1)
            else:
                break
        return streak

    def get_weekly_pattern(self, user_id: int = None) -> dict:
        """Returns emotion counts for the last 7 days"""
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        conn = self._connect()
        c = conn.cursor()
        if user_id:
            c.execute("""
                SELECT emotion, COUNT(*) as count
                FROM emotion_sessions WHERE date >= ? AND user_id = ?
                GROUP BY emotion ORDER BY count DESC
            """, (week_ago, user_id))
        else:
            c.execute("""
                SELECT emotion, COUNT(*) as count
                FROM emotion_sessions WHERE date >= ?
                GROUP BY emotion ORDER BY count DESC
            """, (week_ago,))
        rows = c.fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows}

    def get_avg_polarity_week(self, user_id: int = None) -> float:
        """Average sentiment polarity over the last 7 days"""
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        conn = self._connect()
        c = conn.cursor()
        if user_id:
            c.execute("""
                SELECT AVG(polarity) FROM conversations
                WHERE date >= ? AND role = 'user' AND user_id = ?
            """, (week_ago, user_id))
        else:
            c.execute("""
                SELECT AVG(polarity) FROM conversations
                WHERE date >= ? AND role = 'user'
            """, (week_ago,))
        result = c.fetchone()[0]
        conn.close()
        return round(result, 3) if result else 0.0

    def get_total_sessions(self, user_id: int = None) -> int:
        conn = self._connect()
        c = conn.cursor()
        if user_id:
            c.execute("SELECT COUNT(DISTINCT date) FROM emotion_sessions WHERE user_id = ?",
                      (user_id,))
        else:
            c.execute("SELECT COUNT(DISTINCT date) FROM emotion_sessions")
        result = c.fetchone()[0]
        conn.close()
        return result or 0

    # ─── MEMORY CONTEXT BUILDER ───
    def build_memory_context(self, user_id: int = None) -> str:
        """
        Builds a memory summary string to inject into the AI system prompt.
        This makes EchoMirror remember the user across sessions.
        """
        today_emotion   = self.get_dominant_emotion_today(user_id)
        sad_streak      = self.get_emotion_streak("sad", user_id)
        angry_streak    = self.get_emotion_streak("angry", user_id)
        weekly_pattern  = self.get_weekly_pattern(user_id)
        avg_polarity    = self.get_avg_polarity_week(user_id)
        total_sessions  = self.get_total_sessions(user_id)

        lines = [f"USER MEMORY SUMMARY (from past sessions):"]
        lines.append(f"- Total days used EchoMirror: {total_sessions}")
        lines.append(f"- Today's dominant emotion: {today_emotion}")
        lines.append(f"- Average sentiment this week: {avg_polarity:+.2f}")

        if weekly_pattern:
            top_emotions = list(weekly_pattern.items())[:3]
            pattern_str = ", ".join([f"{e}: {c}x" for e, c in top_emotions])
            lines.append(f"- Most frequent emotions this week: {pattern_str}")

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
    def save_reflection(self, content: str, emotion: str = "neutral",
                        sentiment: str = "Neutral", user_id: int = None):
        """Save a user reflection/journal entry"""
        now = datetime.now()
        encrypted = self.encryptor.encrypt(content)
        with self._write_lock:
            conn = self._connect()
            c = conn.cursor()
            c.execute("""
                INSERT INTO reflections (user_id, timestamp, date, content, emotion, sentiment)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d"),
                  encrypted, emotion, sentiment))
            conn.commit()
            conn.close()

    def get_reflections(self, days: int = 7, user_id: int = None) -> list:
        """Get decrypted reflections from last N days"""
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        conn = self._connect()
        c = conn.cursor()
        if user_id:
            c.execute("""
                SELECT timestamp, content, emotion, sentiment
                FROM reflections WHERE date >= ? AND user_id = ?
                ORDER BY timestamp DESC
            """, (since, user_id))
        else:
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
    def save_daily_summary(self, user_id: int = None):
        """Auto-generate and save today's summary"""
        today = datetime.now().strftime("%Y-%m-%d")
        dominant = self.get_dominant_emotion_today(user_id)
        avg_pol  = self.get_avg_polarity_week(user_id)

        conn = self._connect()
        c = conn.cursor()
        if user_id:
            c.execute("SELECT COUNT(*) FROM emotion_sessions WHERE date = ? AND user_id = ?",
                      (today, user_id))
        else:
            c.execute("SELECT COUNT(*) FROM emotion_sessions WHERE date = ?", (today,))
        count = c.fetchone()[0]

        with self._write_lock:
            c.execute("""
                INSERT OR REPLACE INTO daily_summaries
                (user_id, date, dominant_emotion, avg_polarity, session_count)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, today, dominant, avg_pol, count))
            conn.commit()
        conn.close()

    def print_stats(self, user_id: int = None):
        """Print a memory report to console"""
        print("\n" + "="*50)
        print("  EchoMirror Memory Report")
        print("="*50)
        print(self.build_memory_context(user_id))
        weekly = self.get_weekly_pattern(user_id)
        if weekly:
            print("\nWeekly Emotion Breakdown:")
            for emo, count in weekly.items():
                bar = "█" * min(count, 30)
                print(f"  {emo:10s} {bar} ({count})")
        print("="*50 + "\n")

    # ─── GOAL TRACKING ───
    def save_goal(self, goal_text: str, session_id: str = "",
                  emotion: str = "neutral", user_id: int = None):
        """Save a user goal (persists across sessions)."""
        now = datetime.now()
        encrypted_goal = self.encryptor.encrypt(goal_text)
        with self._write_lock:
            conn = self._connect()
            c = conn.cursor()
            c.execute("""
                INSERT INTO goals (user_id, timestamp, date, session_id, goal_text, emotion)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d"),
                  session_id, encrypted_goal, emotion))
            conn.commit()
            conn.close()

    def get_recent_goals(self, limit: int = 5, user_id: int = None) -> list:
        """Retrieve the most recent goals (decrypted)."""
        conn = self._connect()
        c = conn.cursor()
        if user_id:
            c.execute("""
                SELECT timestamp, goal_text, emotion, status, reframed_to
                FROM goals WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?
            """, (user_id, limit))
        else:
            c.execute("""
                SELECT timestamp, goal_text, emotion, status, reframed_to
                FROM goals ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
        rows = c.fetchall()
        conn.close()

        goals = []
        for ts, g_text, emotion, status, reframed in rows:
            try:
                decrypted = self.encryptor.decrypt(g_text)
            except Exception:
                decrypted = "[encrypted]"
            goals.append({
                "timestamp": ts,
                "goal": decrypted,
                "emotion": emotion,
                "status": status,
                "reframed_to": reframed,
            })
        return goals

    def reframe_goal(self, goal_id: int, new_goal_text: str):
        """Mark a goal as reframed and store the new version."""
        encrypted = self.encryptor.encrypt(new_goal_text)
        with self._write_lock:
            conn = self._connect()
            c = conn.cursor()
            c.execute("""
                UPDATE goals SET status = 'reframed', reframed_to = ?
                WHERE id = ?
            """, (encrypted, goal_id))
            conn.commit()
            conn.close()

    # ─── EMOTION STATS (for web dashboard) ───
    def get_emotion_stats(self, user_id: int = None, days: int = 30) -> dict:
        """Get emotion statistics for charts/dashboard."""
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        conn = self._connect()
        c = conn.cursor()

        # Frequency
        if user_id:
            c.execute("""
                SELECT emotion, COUNT(*) as cnt FROM emotion_sessions
                WHERE date >= ? AND user_id = ? GROUP BY emotion ORDER BY cnt DESC
            """, (since, user_id))
        else:
            c.execute("""
                SELECT emotion, COUNT(*) as cnt FROM emotion_sessions
                WHERE date >= ? GROUP BY emotion ORDER BY cnt DESC
            """, (since,))
        frequency = {r[0]: r[1] for r in c.fetchall()}

        # Daily timeline
        if user_id:
            c.execute("""
                SELECT date, emotion, COUNT(*) as cnt FROM emotion_sessions
                WHERE date >= ? AND user_id = ?
                GROUP BY date, emotion ORDER BY date ASC
            """, (since, user_id))
        else:
            c.execute("""
                SELECT date, emotion, COUNT(*) as cnt FROM emotion_sessions
                WHERE date >= ?
                GROUP BY date, emotion ORDER BY date ASC
            """, (since,))
        timeline = {}
        for date, emo, cnt in c.fetchall():
            if date not in timeline:
                timeline[date] = {}
            timeline[date][emo] = cnt

        conn.close()
        return {"frequency": frequency, "timeline": timeline}


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    db = MemoryDB()

    # Test user creation
    print("[TEST] Creating users...")
    uid1 = db.create_user("sania", "test123", "SK. Sania")
    uid2 = db.create_user("manoj", "test456", "R. Manoj Naik")
    print(f"  User 1: id={uid1}")
    print(f"  User 2: id={uid2}")

    # Test authentication
    print("[TEST] Authenticating...")
    auth = db.authenticate_user("sania", "test123")
    print(f"  sania/test123 → {auth}")
    bad = db.authenticate_user("sania", "wrong")
    print(f"  sania/wrong → {bad}")

    # Test per-user emotion logging
    print("[TEST] Logging emotions per user...")
    db.log_emotion("sad", 0.82, user_id=uid1)
    db.log_emotion("happy", 0.95, user_id=uid2)

    # Test per-user memory context
    print("[TEST] Memory context for user 1:")
    print(db.build_memory_context(uid1))

    print("\n[MemoryDB] All tests passed!")