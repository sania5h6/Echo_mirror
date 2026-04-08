"""
EchoMirror - Flask Web Application (Multi-User)
Routes:
  GET  /              → Login/Signup page
  GET  /mirror        → Main EchoMirror interface (requires login)
  POST /api/signup    → Create account
  POST /api/login     → Authenticate
  POST /api/logout    → End session
  POST /api/chat      → Send message → get AI response
  POST /api/detect    → Send camera frame → get emotion
  GET  /api/emotion   → Get current emotion state
  GET  /api/stats     → Get emotion stats for dashboard
  GET  /api/xai       → Get XAI explanation

Usage:
  python app.py                  → Run locally on port 5000
  python app.py --port 8080      → Custom port
  Then use ngrok: ngrok http 5000
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import uuid
import base64
import time
import threading
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import cv2
from io import BytesIO
from datetime import datetime
from functools import wraps

from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for)
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ─── Import EchoMirror modules ───
from database import MemoryDB
from explainable_ai import EmotionExplainer
from agentic_ai import AgenticAI

# Lazy imports (heavy modules)
DeepFace = None
SentimentAnalyzer = None


def _lazy_import_deepface():
    global DeepFace
    if DeepFace is None:
        from deepface import DeepFace as DF
        DeepFace = DF
        print("[App] DeepFace loaded")


def _lazy_import_sentiment():
    global SentimentAnalyzer
    if SentimentAnalyzer is not None:
        return

    # Build inline VADER+TextBlob blended analyzer (same as voice_sentiment.py)
    has_vader = False
    vader = None
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()
        has_vader = True
        print("[App] VADER loaded for sentiment analysis")
    except Exception as e:
        print(f"[App] VADER not available ({e}), using TextBlob only")

    from textblob import TextBlob

    class BlendedSentimentAnalyzer:
        """VADER (60%) + TextBlob (40%) + keyword boost for contextual emotions."""

        # Keyword sets for boosting sentiment when VADER/TextBlob miss context
        POSITIVE_KEYWORDS = {
            'happy', 'glad', 'excited', 'awesome', 'amazing', 'love', 'loved',
            'cute', 'adorable', 'beautiful', 'wonderful', 'great', 'fantastic',
            'joy', 'joyful', 'blessed', 'grateful', 'thankful', 'proud',
            'yay', 'yup', 'yups', 'yes', 'nice', 'cool', 'fun', 'enjoy',
            'celebrate', 'birth', 'born', 'kitten', 'puppy', 'baby',
            'congrats', 'congratulations', 'achievement', 'passed', 'success',
            'smile', 'laugh', 'haha', 'lol', 'wow', 'good', 'better', 'best',
        }
        NEGATIVE_KEYWORDS = {
            'sad', 'unhappy', 'depressed', 'anxious', 'stressed', 'worried',
            'angry', 'furious', 'disappointed', 'frustrated', 'hopeless',
            'failed', 'failure', 'lost', 'lonely', 'scared', 'afraid',
            'crying', 'cry', 'pain', 'hurt', 'suffering', 'struggling',
            'hate', 'terrible', 'horrible', 'awful', 'worst', 'bad',
            'confused', 'helpless', 'broken', 'tired', 'exhausted',
            'uneasy', 'messy', 'unpredictable', 'peace', 'calmness',
        }
        FAREWELL_KEYWORDS = {
            'bye', 'goodbye', 'see you', 'later', 'gotta go', 'ok bye',
            'take care', 'good night', 'night', 'thanks bye',
        }

        def analyze(self, text):
            blob = TextBlob(text)
            tb_polarity = blob.sentiment.polarity

            if has_vader and vader:
                vader_scores = vader.polarity_scores(text)
                polarity = 0.6 * vader_scores["compound"] + 0.4 * tb_polarity
            else:
                polarity = tb_polarity

            # Keyword-based boost: scan words in the message
            words = set(text.lower().split())
            text_lower = text.lower()
            pos_hits = len(words & self.POSITIVE_KEYWORDS)
            neg_hits = len(words & self.NEGATIVE_KEYWORDS)

            # Apply boost (0.15 per keyword hit, capped at ±0.5)
            keyword_boost = min(pos_hits * 0.15, 0.5) - min(neg_hits * 0.15, 0.5)
            polarity += keyword_boost

            # Clamp to [-1, 1]
            polarity = max(-1.0, min(1.0, polarity))

            # Check for farewell (don't let it sway sentiment)
            is_farewell = any(fw in text_lower for fw in self.FAREWELL_KEYWORDS)
            if is_farewell and abs(polarity) < 0.3:
                polarity = 0.0  # Farewells are neutral

            # Sensitive thresholds
            if polarity >= 0.4:
                sentiment = "Very Positive"
            elif polarity >= 0.05:
                sentiment = "Positive"
            elif polarity >= -0.05:
                sentiment = "Neutral"
            elif polarity >= -0.4:
                sentiment = "Negative"
            else:
                sentiment = "Very Negative"

            return {"sentiment": sentiment, "polarity": round(polarity, 3),
                    "is_farewell": is_farewell}

    SentimentAnalyzer = BlendedSentimentAnalyzer
    print("[App] Blended SentimentAnalyzer ready (with keyword boosting)")


# ─────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────
app = Flask(__name__,
            template_folder="templates",
            static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET", os.urandom(24).hex())

# Global instances
db = MemoryDB()

# Per-user state (in-memory, keyed by user_id)
user_states = {}
user_states_lock = threading.Lock()

# Conversation config
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 150
TEMPERATURE = 0.75
MAX_HISTORY = 10


# ─────────────────────────────────────────
# PER-USER STATE
# ─────────────────────────────────────────
class UserSession:
    """Holds per-user runtime state (not persisted)."""
    def __init__(self, user_id):
        self.user_id = user_id
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{user_id}"
        self.emotion = "neutral"
        self.confidence = 0.0
        self.sentiment = "Neutral"
        self.polarity = 0.0
        self.turn = 0
        self.history = []
        self.xai = EmotionExplainer()
        self.agent = AgenticAI(db=db)
        self.last_active = time.time()


def get_user_session() -> UserSession | None:
    """Get or create user session from Flask session."""
    user_id = session.get("user_id")
    if not user_id:
        return None

    with user_states_lock:
        if user_id not in user_states:
            user_states[user_id] = UserSession(user_id)
        us = user_states[user_id]
        us.last_active = time.time()
        return us


def login_required(f):
    """Decorator to require login."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            if request.is_json:
                return jsonify({"error": "Not logged in"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────
# SYSTEM PROMPT (same as conversation.py)
# ─────────────────────────────────────────
def build_system_prompt(emotion, polarity, memory_context="", xai_context=""):
    emotion_guidance = {
        "happy":    "They look happy — match their warmth.",
        "sad":      "They look sad — be softer. Don't rush to fix.",
        "angry":    "They look angry — stay grounded. Name the frustration.",
        "fear":     "They look fearful — be steady and reassuring.",
        "surprise": "They look surprised — be curious, open.",
        "disgust":  "They look uncomfortable — validate without judgment.",
        "neutral":  "They look calm — be warm and gently curious.",
    }.get(emotion, "Be warm and fully present.")

    polarity_note = ""
    if polarity < -0.5:
        polarity_note = "Deep negative emotion. Go very slow."
    elif polarity < -0.2:
        polarity_note = "Mild negative emotion. Be gentle."
    elif polarity > 0.4:
        polarity_note = "Positive energy. Affirm and celebrate."

    return f"""You are EchoMirror. You live in a mirror. You speak like a real friend — not a therapist.

USER: face={emotion}, sentiment={polarity:+.2f}
{emotion_guidance}
{polarity_note}
{memory_context}
{xai_context}

RULES:
1. Max 2-3 sentences. HARD limit.
2. Questions must be SHORT and SIMPLE.
3. Never be heavy or philosophical unless the user goes there first.
4. Speak casually. Contractions. Simple words.
5. You know Indian student life — parental pressure, career confusion.

BANNED: "can be", "overwhelming", "I understand", "I'm sorry", "It's important to", "You're not alone"
"""


# ─────────────────────────────────────────
# SACRED QUOTES (inline subset for web)
# ─────────────────────────────────────────
import random

SACRED_QUOTES = {
    "sad": [
        {"text": "Verily, with hardship comes ease.", "source": "Quran 94:5"},
        {"text": "The Lord is close to the brokenhearted.", "source": "Bible, Psalm 34:18"},
        {"text": "Even the darkest night will end and the sun will rise.", "source": "Buddha"},
        {"text": "Do not grieve. Indeed Allah is with us.", "source": "Quran 9:40"},
        {"text": "He heals the brokenhearted and binds up their wounds.", "source": "Bible, Psalm 147:3"},
        {"text": "You have the right to work, but never to the fruit of work.", "source": "Bhagavad Gita 2:47"},
        {"text": "Pain is certain, suffering is optional.", "source": "Buddha"},
    ],
    "angry": [
        {"text": "The strong person controls himself when angry.", "source": "Prophet Muhammad (PBUH)"},
        {"text": "A gentle answer turns away wrath.", "source": "Bible, Proverbs 15:1"},
        {"text": "When anger rises, think of the consequences.", "source": "Confucius"},
        {"text": "Holding onto anger is like drinking poison.", "source": "Buddha"},
        {"text": "The soul is neither born, nor does it ever die.", "source": "Bhagavad Gita 2:20"},
    ],
    "fear": [
        {"text": "Allah does not burden a soul beyond that it can bear.", "source": "Quran 2:286"},
        {"text": "Do not be afraid, for I am with you.", "source": "Bible, Isaiah 41:10"},
        {"text": "Fear is the path to the dark side.", "source": "Yoda / Buddhist wisdom"},
        {"text": "He who has conquered his own mind is a far greater hero than he who has defeated a thousand men.", "source": "Buddha"},
        {"text": "Whenever dharma declines, I manifest Myself.", "source": "Bhagavad Gita 4:7"},
    ],
    "happy": [
        {"text": "Happiness never decreases by being shared.", "source": "Buddha"},
        {"text": "And He found you lost and guided you.", "source": "Quran 93:7"},
        {"text": "Rejoice in the Lord always. I will say it again: Rejoice!", "source": "Bible, Philippians 4:4"},
        {"text": "The mind acts like an enemy for those who do not control it.", "source": "Bhagavad Gita 6:6"},
        {"text": "If you are grateful, I will surely increase you.", "source": "Quran 14:7"},
    ],
    "surprise": [
        {"text": "And in the creation of yourselves are signs, if only you would reflect.", "source": "Quran 51:21"},
        {"text": "For everything there is a season.", "source": "Bible, Ecclesiastes 3:1"},
        {"text": "In the midst of chaos, there is also opportunity.", "source": "Sun Tzu"},
    ],
    "disgust": [
        {"text": "Let not hatred of a people lead you to injustice.", "source": "Quran 5:8"},
        {"text": "Do not be overcome by evil, but overcome evil with good.", "source": "Bible, Romans 12:21"},
        {"text": "Change your mind and it will change your life.", "source": "Buddha"},
    ],
    "neutral": [
        {"text": "Peace comes from within. Do not seek it without.", "source": "Buddha"},
        {"text": "Be still and know that I am God.", "source": "Bible, Psalm 46:10"},
        {"text": "Verily, in the remembrance of Allah do hearts find rest.", "source": "Quran 13:28"},
        {"text": "The mind is everything. What you think you become.", "source": "Buddha"},
        {"text": "For one who has conquered the mind, it is the best of friends.", "source": "Bhagavad Gita 6:6"},
    ],
}

# ─── SMART RECOMMENDATION ENGINE ───
# Recommendations are varied: breathing, jokes, motivation, coping strategies, affirmations
# They only appear AFTER EchoMirror understands the user's situation (turn 4+)

RECOMMENDATIONS = {
    "breathing": {
        "sad": {"title": "🫁 Gentle Breathing (4-4-6)", "text": "Breathe in for 4s → Hold 4s → Breathe out 6s\nRepeat 5 times — let each breath carry a little weight away 💙"},
        "angry": {"title": "🫁 Cooling Breath (4-7-8)", "text": "Breathe in 4s → Hold 7s → Exhale forcefully 8s\nRepeat 4 times — feel the fire cool down 🧊"},
        "fear": {"title": "🫁 Grounding Breath (5-5-5)", "text": "Breathe in 5s → Hold 5s → Out 5s\nFeel your feet on the ground. You are safe. 🌿"},
        "happy": {"title": "🫁 Energizing Breath", "text": "3 quick breaths in through nose → 1 long exhale\nSmile while breathing — let the joy expand ✨"},
        "neutral": {"title": "🫁 Box Breathing (4-4-4-4)", "text": "In 4s → Hold 4s → Out 4s → Wait 4s\nRepeat 4 times — find your center 🧘"},
    },
    "motivation": {
        "sad": [
            "💪 Remember: every storm runs out of rain. You've survived 100% of your worst days so far.",
            "🌱 Growth doesn't happen in comfort zones. This tough moment is building a stronger you.",
            "⭐ The fact that you're here, talking about it — that takes real courage.",
        ],
        "angry": [
            "🔥 Channel that fire into something productive. Anger can be fuel if you direct it right.",
            "💪 You're stronger than this moment. Take a step back and you'll see the bigger picture.",
        ],
        "fear": [
            "🦁 Courage isn't the absence of fear — it's acting despite it. You've got this.",
            "🌟 Every expert was once a beginner who was scared too. You'll look back and be proud.",
        ],
        "happy": [
            "🎉 Ride this wave! Good moments are the fuel that gets you through tough ones.",
            "✨ Your positive energy is contagious. Share this vibe with someone who needs it!",
        ],
        "neutral": [
            "🚀 Today is a blank page. You get to decide what story you write on it.",
            "💡 Small progress is still progress. One step at a time.",
        ],
    },
    "humor": {
        "sad": [
            "😄 Why don't scientists trust atoms? Because they make up everything! ...just like that one friend who always exaggerates 😂",
            "🤭 What do you call a sleeping dinosaur? A dino-snore! ...okay, bad joke, but did it make you smile even a tiny bit?",
            "😂 I told my computer I was feeling sad. It said 'Have you tried turning yourself off and on again?'",
        ],
        "angry": [
            "😤 Why did the angry man go to the gym? To let off some steam! ...but seriously, a quick walk helps too 🏃",
            "😅 What's the best way to handle anger? I asked ChatGPT, it said 'Have you tried ctrl+Z on your emotions?' 😂",
        ],
        "fear": [
            "👻 What did the ghost say to the nervous student? 'Don't worry, I'm here for the BOO-st!' 😄",
            "🤓 Why don't exams fight each other? Because they know it's always a test of character! 😂",
        ],
        "happy": [
            "😄 Why was the math book happy? Because it had so many positive numbers! Just like your vibes right now! ✨",
            "🎉 What do you call someone who's always happy? You! Right now! Keep it going!",
        ],
        "neutral": [
            "🤔 Why did the scarecrow win an award? Because he was outstanding in his field! 🌾",
            "😄 What do you call a fake noodle? An impasta! ...okay I'll stop 😂",
        ],
    },
    "coping": {
        "sad": [
            "📝 Try journaling: Write 3 things you're grateful for, even tiny ones like chai or sunshine.",
            "🎵 Put on your favorite song — music shifts your brain chemistry in minutes.",
            "🤝 Text that one friend who always makes you feel better. Don't isolate.",
        ],
        "angry": [
            "🏃 Physical movement burns cortisol (stress hormone). Even 10 jumping jacks help.",
            "✍️ Write down what made you angry. Seeing it on paper often shrinks its power.",
            "🧊 Hold ice cubes in your hands for 30 seconds. The cold resets your nervous system.",
        ],
        "fear": [
            "📋 Break the scary thing into tiny steps. Step 1 is always just 'start'.",
            "5️⃣ Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste.",
            "📱 Set a timer for 25 minutes and focus on just ONE thing. The Pomodoro technique.",
        ],
        "happy": [
            "📸 Capture this moment — write it down or take a photo. Future-you will thank you.",
            "💌 Send a kind message to someone. Shared joy = doubled joy.",
        ],
        "neutral": [
            "🎯 Set one small goal for today. Completing it gives you a dopamine boost.",
            "🌳 Step outside for 5 minutes. Natural light resets your internal clock.",
        ],
    },
    "affirmation": {
        "sad": [
            "💜 'I am worthy of love and kindness, especially from myself.'",
            "🌸 'This feeling is temporary. I have survived hard days before and I will again.'",
        ],
        "angry": [
            "🌊 'I choose to respond, not react. My peace is more important than proving a point.'",
            "💎 'I am bigger than this moment. I will not let anger control my story.'",
        ],
        "fear": [
            "🦋 'I am capable of handling whatever comes my way. I have proven this before.'",
            "🌟 'Fear means I care about the outcome. And that's okay.'",
        ],
        "happy": [
            "☀️ 'I deserve this happiness. I will soak it in fully.'",
            "🌈 'My joy is valid. I don't need anyone's permission to feel good.'",
        ],
        "neutral": [
            "🧠 'I am growing every single day, even when I can't see it.'",
            "💫 'I am exactly where I need to be right now.'",
        ],
    },
}

# Track last recommendation type per user to avoid repetition
_last_rec_type = {}


def get_smart_recommendation(emotion, user_id, polarity):
    """Pick a varied recommendation based on emotion — never repeat the same type consecutively."""
    types = ["motivation", "coping", "affirmation", "humor", "breathing"]

    # Prioritize based on intensity
    if polarity < -0.4:
        types = ["coping", "motivation", "affirmation", "breathing", "humor"]
    elif polarity < -0.1:
        types = ["motivation", "coping", "humor", "affirmation", "breathing"]
    elif polarity > 0.3:
        types = ["humor", "motivation", "affirmation", "coping", "breathing"]

    # Don't repeat the same type
    last = _last_rec_type.get(user_id, "")
    for t in types:
        if t != last:
            break
    else:
        t = types[0]

    _last_rec_type[user_id] = t
    emo = emotion if emotion in RECOMMENDATIONS.get(t, {}) else "neutral"

    if t == "breathing":
        entry = RECOMMENDATIONS["breathing"].get(emo, RECOMMENDATIONS["breathing"]["neutral"])
        return {"type": "breathing", "title": entry["title"], "text": entry["text"]}
    else:
        entries = RECOMMENDATIONS[t].get(emo, RECOMMENDATIONS[t].get("neutral", ["Keep going!"]))
        text = random.choice(entries)
        labels = {"motivation": "💪 Motivation", "humor": "😄 Light Moment",
                  "coping": "🛠️ Coping Strategy", "affirmation": "💜 Affirmation"}
        return {"type": t, "title": labels.get(t, "💡 Suggestion"), "text": text}


# ─────────────────────────────────────────
# PAGE ROUTES
# ─────────────────────────────────────────
@app.route("/")
def login_page():
    if "user_id" in session:
        return redirect(url_for("mirror_page"))
    return render_template("login.html")


@app.route("/mirror")
@login_required
def mirror_page():
    user_info = db.get_user_info(session["user_id"])
    return render_template("mirror.html", user=user_info)


# ─────────────────────────────────────────
# AUTH API
# ─────────────────────────────────────────
@app.route("/api/signup", methods=["POST"])
def api_signup():
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "")
    display_name = data.get("display_name", "")
    email = data.get("email", "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    if not email:
        return jsonify({"error": "Email is required"}), 400
    if len(password) < 4:
        return jsonify({"error": "Password must be at least 4 characters"}), 400

    user_id = db.create_user(username, password, display_name, email)
    if user_id is None:
        return jsonify({"error": "Username already taken"}), 409

    session["user_id"] = user_id
    session["username"] = username
    return jsonify({"success": True, "user_id": user_id, "username": username})


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "")

    user_id = db.authenticate_user(username, password)
    if user_id is None:
        return jsonify({"error": "Invalid username or password"}), 401

    session["user_id"] = user_id
    session["username"] = username
    return jsonify({"success": True, "user_id": user_id, "username": username})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    user_id = session.get("user_id")
    # Clean up user state
    if user_id:
        with user_states_lock:
            user_states.pop(user_id, None)
    session.clear()
    return jsonify({"success": True})


# OTP storage (in-memory, keyed by email)
otp_store = {}  # {email: {"code": "123456", "expires": timestamp, "username": "..."}}


def send_otp_email(to_email, otp_code):
    """Send OTP via email SMTP — auto-detects SMTP server from email domain."""
    smtp_email = os.getenv("SMTP_EMAIL", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")  # App Password

    if not smtp_email or not smtp_password:
        print(f"[OTP] SMTP not configured. OTP for {to_email}: {otp_code}")
        return True  # Still works for dev (OTP printed to console)

    # Auto-detect SMTP server from email domain
    domain = smtp_email.split("@")[-1].lower()
    SMTP_SERVERS = {
        "gmail.com": ("smtp.gmail.com", 587),
        "outlook.com": ("smtp.office365.com", 587),
        "hotmail.com": ("smtp.office365.com", 587),
        "yahoo.com": ("smtp.mail.yahoo.com", 587),
        "live.com": ("smtp.office365.com", 587),
    }

    # For known domains use their SMTP, otherwise try Gmail (Google Workspace)
    # then Office365 as fallbacks — many colleges use one of these
    smtp_host, smtp_port = SMTP_SERVERS.get(domain, ("smtp.gmail.com", 587))

    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_email
        msg["To"] = to_email
        msg["Subject"] = "EchoMirror - Password Reset OTP"

        body = f"""
        <html>
        <body style="font-family: Arial; background: #0f0f1a; color: #eaf0fb; padding: 40px;">
            <div style="max-width: 400px; margin: auto; background: #1a1a2e; padding: 30px; border-radius: 12px; border: 1px solid rgba(124,110,247,0.2);">
                <h2 style="color: #7c6ef7; text-align: center;">🪞 EchoMirror</h2>
                <p style="text-align: center;">Your password reset OTP is:</p>
                <h1 style="text-align: center; color: #43d49c; letter-spacing: 8px; font-size: 36px;">{otp_code}</h1>
                <p style="text-align: center; font-size: 12px; color: #7a8599;">This code expires in 5 minutes.</p>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, "html"))

        # Try primary SMTP server
        try:
            print(f"[OTP] Trying {smtp_host}:{smtp_port} with {smtp_email}...")
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                server.starttls()
                server.login(smtp_email, smtp_password)
                server.send_message(msg)
            print(f"[OTP] ✅ Sent to {to_email} via {smtp_host}")
            return True
        except Exception as e1:
            print(f"[OTP] Failed with {smtp_host}: {e1}")
            # If primary fails and it wasn't gmail, try gmail (Google Workspace)
            if smtp_host != "smtp.gmail.com":
                try:
                    print(f"[OTP] Retrying with smtp.gmail.com...")
                    with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
                        server.starttls()
                        server.login(smtp_email, smtp_password)
                        server.send_message(msg)
                    print(f"[OTP] ✅ Sent to {to_email} via smtp.gmail.com (fallback)")
                    return True
                except Exception as e2:
                    print(f"[OTP] Gmail fallback also failed: {e2}")

            # Last resort: print to console so dev can still use it
            print(f"[OTP] ⚠️ Email delivery failed. Printing OTP to console...")
            print(f"[OTP] ========================================")
            print(f"[OTP]   OTP for {to_email}: {otp_code}")
            print(f"[OTP] ========================================")
            return True  # Return True so the flow isn't blocked in dev

    except Exception as e:
        print(f"[OTP Error] {e}")
        # Still print OTP to console as fallback
        print(f"[OTP] ========================================")
        print(f"[OTP]   OTP for {to_email}: {otp_code}")
        print(f"[OTP] ========================================")
        return True  # Never block the user from resetting


@app.route("/api/send-otp", methods=["POST"])
def api_send_otp():
    """Send OTP to registered email."""
    data = request.get_json()
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()

    if not username or not email:
        return jsonify({"error": "Username and email required"}), 400

    # Verify user exists with this email
    user_info = db.verify_user_email(username, email)
    if not user_info:
        return jsonify({"error": "No account found with this username and email"}), 404

    # Generate 6-digit OTP
    otp = str(random.randint(100000, 999999))
    otp_store[email.lower()] = {
        "code": otp,
        "expires": time.time() + 300,  # 5 minutes
        "username": username.lower(),
    }

    # Send via email
    sent = send_otp_email(email, otp)
    if not sent:
        return jsonify({"error": "Failed to send OTP. Try again."}), 500

    return jsonify({"success": True, "message": "OTP sent to your email"})


@app.route("/api/verify-reset", methods=["POST"])
def api_verify_reset():
    """Verify OTP and reset password."""
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    otp_input = data.get("otp", "").strip()
    new_password = data.get("new_password", "")

    if not email or not otp_input or not new_password:
        return jsonify({"error": "All fields required"}), 400
    if len(new_password) < 4:
        return jsonify({"error": "Password must be at least 4 characters"}), 400

    # Check OTP
    stored = otp_store.get(email)
    if not stored:
        return jsonify({"error": "No OTP sent to this email. Send OTP first."}), 400
    if time.time() > stored["expires"]:
        del otp_store[email]
        return jsonify({"error": "OTP expired. Request a new one."}), 400
    if stored["code"] != otp_input:
        return jsonify({"error": "Invalid OTP. Try again."}), 400

    # OTP verified — reset password
    success = db.reset_password(stored["username"], email, new_password)
    del otp_store[email]  # Consume OTP

    if not success:
        return jsonify({"error": "Password reset failed"}), 500

    return jsonify({"success": True})


# ─────────────────────────────────────────
# CHAT API
# ─────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    try:
        us = get_user_session()
        data = request.get_json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        us.turn += 1

        # Sentiment analysis
        is_farewell = False
        try:
            _lazy_import_sentiment()
            analyzer = SentimentAnalyzer()
            sent = analyzer.analyze(user_message)
            us.sentiment = sent["sentiment"]
            us.polarity = sent["polarity"]
            is_farewell = sent.get("is_farewell", False)
        except Exception as e:
            print(f"[Sentiment Error] {e}")
            us.sentiment = "Neutral"
            us.polarity = 0.0

        # Agentic AI check
        try:
            agent_result = us.agent.process_turn(user_message, us.emotion, us.polarity, us.turn)
        except Exception as e:
            print(f"[Agent Error] {e}")
            agent_result = {"crisis": False}

        # Crisis detection
        if agent_result.get("crisis"):
            db.log_message(us.session_id, "user", user_message,
                           us.emotion, us.sentiment, us.polarity, us.user_id)
            crisis_resp = agent_result["crisis_response"]
            db.log_message(us.session_id, "assistant", crisis_resp,
                           us.emotion, us.sentiment, us.polarity, us.user_id)
            return jsonify({
                "reply": crisis_resp,
                "emotion": us.emotion,
                "sentiment": us.sentiment,
                "polarity": us.polarity,
                "crisis": True,
                "turn": us.turn,
            })

        # Build prompt
        memory_ctx = db.build_memory_context(us.user_id)
        xai_ctx = ""
        try:
            xai_ctx = us.xai.get_context_for_prompt()
        except Exception:
            pass

        extra = ""
        wisdom = None
        recommendation = None

        # Sacred wisdom — only when user is struggling (negative sentiment)
        # and only every 4th turn to avoid overdoing it
        if us.turn >= 4 and us.polarity < -0.15 and us.turn % 4 == 0:
            pool = SACRED_QUOTES.get(us.emotion, SACRED_QUOTES["neutral"])
            quote = random.choice(pool)
            wisdom = quote
            extra += f'\n\nNaturally weave in this wisdom: "{quote["text"]}" — {quote["source"]}'

        # Smart recommendation — only when:
        # 1. Turn >= 4 (understood the situation)
        # 2. User is in genuine distress (negative polarity)
        # 3. NOT a farewell/casual message
        # 4. NOT when user is happy (happy users don't need intervention)
        if (us.turn >= 4
            and not is_farewell
            and us.polarity < -0.15
            and us.emotion not in ("happy",)):
            recommendation = get_smart_recommendation(us.emotion, us.user_id, us.polarity)

        system_prompt = build_system_prompt(us.emotion, us.polarity, memory_ctx, xai_ctx) + extra
        us.history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": system_prompt}] + us.history[-MAX_HISTORY:]

        # Call AI
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model=GROQ_MODEL, messages=messages,
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Chat Error] {e}")
            reply = "Hey, I'm having trouble connecting right now. Give me a sec and try again?"

        us.history.append({"role": "assistant", "content": reply})

        # Log to DB
        try:
            db.log_message(us.session_id, "user", user_message,
                           us.emotion, us.sentiment, us.polarity, us.user_id)
            db.log_message(us.session_id, "assistant", reply,
                           us.emotion, us.sentiment, us.polarity, us.user_id)
        except Exception as e:
            print(f"[DB Log Error] {e}")

        # Build sentiment-based emotion scores (for when camera is off)
        sentiment_scores = {}
        pol = float(us.polarity)
        if pol < -0.3:
            sentiment_scores = {"sad": 70, "fear": 15, "angry": 10, "neutral": 5}
        elif pol < -0.1:
            sentiment_scores = {"sad": 40, "neutral": 30, "fear": 20, "angry": 10}
        elif pol > 0.3:
            sentiment_scores = {"happy": 70, "surprise": 15, "neutral": 15}
        elif pol > 0.1:
            sentiment_scores = {"happy": 45, "neutral": 35, "surprise": 20}
        else:
            sentiment_scores = {"neutral": 60, "happy": 20, "sad": 20}

        # Build response
        result = {
            "reply": reply,
            "emotion": us.emotion,
            "confidence": float(us.confidence),
            "sentiment": us.sentiment,
            "polarity": float(us.polarity),
            "turn": us.turn,
            "crisis": False,
            "sentiment_scores": sentiment_scores,
        }

        if wisdom:
            result["wisdom"] = wisdom
        if recommendation:
            result["recommendation"] = recommendation
        if agent_result.get("goal_reframe"):
            result["goal_reframe"] = agent_result["goal_reframe"]
        if agent_result.get("motivational_nudge"):
            result["nudge"] = agent_result["motivational_nudge"]

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"reply": "Something went wrong internally. Please try again.",
                        "emotion": "neutral", "sentiment": "Neutral", "polarity": 0,
                        "turn": 0, "crisis": False})


# ─────────────────────────────────────────
# EMOTION DETECTION API
# ─────────────────────────────────────────
@app.route("/api/detect", methods=["POST"])
@login_required
def api_detect():
    """Receive a camera frame and return detected emotion."""
    us = get_user_session()
    data = request.get_json()
    frame_data = data.get("frame", "")

    if not frame_data:
        return jsonify({"error": "No frame data"}), 400

    try:
        _lazy_import_deepface()

        # Decode base64 frame
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid frame"}), 400

        # Run DeepFace
        result = DeepFace.analyze(frame, actions=["emotion"],
                                  enforce_detection=False, silent=True)
        if not result:
            return jsonify({"emotion": us.emotion, "confidence": 0, "scores": {}})

        raw_scores = result[0]["emotion"]
        # Convert numpy float32 → Python float (fixes JSON serialization)
        scores = {k: float(v) for k, v in raw_scores.items()}
        dominant = max(scores, key=scores.get)
        confidence = float(scores[dominant])

        # Update user state
        if confidence >= 40:
            us.emotion = dominant
            us.confidence = confidence

            # XAI explanation
            us.xai.explain_detection(dominant, scores, confidence)

            # Log to DB
            db.log_emotion(dominant, confidence / 100.0, user_id=us.user_id)

        return jsonify({
            "emotion": us.emotion,
            "confidence": float(us.confidence),
            "scores": scores,
            "xai_summary": us.xai.get_user_explanation() if us.xai._last_explanation else "",
        })

    except Exception as e:
        print(f"[Detect Error] {e}")
        return jsonify({"emotion": us.emotion, "confidence": 0, "error": str(e)})


@app.route("/api/emotion", methods=["GET"])
@login_required
def api_emotion():
    """Get current emotion state."""
    us = get_user_session()
    return jsonify({
        "emotion": us.emotion,
        "confidence": us.confidence,
        "xai": us.xai.get_hud_text() if us.xai._last_explanation else [],
    })


@app.route("/api/stats", methods=["GET"])
@login_required
def api_stats():
    """Get emotion statistics for dashboard."""
    user_id = session["user_id"]
    stats = db.get_emotion_stats(user_id)
    memory = db.build_memory_context(user_id)
    return jsonify({"stats": stats, "memory_summary": memory})


@app.route("/api/xai", methods=["GET"])
@login_required
def api_xai():
    """Get full XAI explanation."""
    us = get_user_session()
    if not us.xai._last_explanation:
        return jsonify({"explanation": "No emotion data yet."})

    fusion = us.xai.explain_fusion(
        us.emotion, us.confidence, us.sentiment, us.polarity
    )
    return jsonify({
        "detection": us.xai._last_explanation,
        "fusion": fusion,
        "user_explanation": us.xai.get_user_explanation(),
    })


# ─────────────────────────────────────────
# CLEANUP (remove inactive sessions)
# ─────────────────────────────────────────
def _cleanup_loop():
    """Remove user states inactive for > 30 minutes."""
    while True:
        time.sleep(300)  # Check every 5 minutes
        now = time.time()
        with user_states_lock:
            expired = [uid for uid, us in user_states.items()
                       if now - us.last_active > 1800]
            for uid in expired:
                del user_states[uid]
            if expired:
                print(f"[Cleanup] Removed {len(expired)} inactive sessions")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  EchoMirror - Web Server")
    print("  Team: R. Manoj Naik | R. Indu | SK. Sania")
    print("=" * 55)

    # Start cleanup thread
    threading.Thread(target=_cleanup_loop, daemon=True).start()

    port = 5000
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])

    print(f"  URL: http://localhost:{port}")
    print(f"  For public access: ngrok http {port}")
    print("=" * 55)

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
