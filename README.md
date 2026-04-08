<p align="center">
  <img src="https://img.shields.io/badge/EchoMirror-AI%20Emotional%20Support-7c6ef7?style=for-the-badge&logo=sparkles" alt="EchoMirror Badge"/>
</p>

<h1 align="center">🪞 EchoMirror</h1>
<h3 align="center">AI-Powered Personal Emotional Reflection & Support System</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-3.x-000000?logo=flask" />
  <img src="https://img.shields.io/badge/DeepFace-Emotion%20Detection-FF6F61" />
  <img src="https://img.shields.io/badge/LLaMA%203.3-70B-43d49c?logo=meta" />
  <img src="https://img.shields.io/badge/License-MIT-blue" />
</p>

---

## 📖 About

**EchoMirror** is an AI-powered web platform that combines **real-time facial emotion detection**, **text/voice sentiment analysis**, and **context-aware conversational AI** to provide personalized emotional support. Unlike traditional chatbots, EchoMirror truly *understands* how you feel — through your face, your words, and your voice.

> *"At every reflection, we heal, we rise, we rediscover you."*

### 🎯 Problem Statement

Mental health challenges among students and young adults are increasing, yet access to immediate emotional support remains limited due to stigma, cost, and availability. Traditional chatbots provide generic, one-size-fits-all responses without understanding the user's actual emotional state. EchoMirror bridges this gap with multi-modal emotion awareness and personalized, empathetic AI companionship.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎭 **Facial Emotion Detection** | Real-time webcam analysis using DeepFace (7 emotions: happy, sad, angry, fear, surprise, disgust, neutral) |
| 💬 **AI Conversation Engine** | Empathetic, multi-turn chat powered by LLaMA 3.3 70B via Groq API |
| 🎙️ **Voice Input** | Speech-to-text using Web Speech API with live sentiment analysis |
| 📊 **Sentiment Analysis** | VADER (60%) + TextBlob (40%) blended analyzer with keyword-based emotion boosting |
| 🧠 **Explainable AI (XAI)** | Transparent explanations of why a specific emotion was detected |
| 🤖 **Agentic AI** | Smart recommendations — breathing exercises, coping strategies, motivational thoughts, humor, affirmations |
| ✝️☪️🕉️☸️ **Sacred Wisdom** | Context-aware quotes from Quran, Bible, Bhagavad Gita, and Buddha |
| 🚨 **Crisis Detection** | Detects self-harm/suicidal ideation and provides helpline resources |
| 🔐 **Secure Auth** | Multi-user accounts with bcrypt password hashing and email OTP password reset |
| 📈 **Emotion Tracking** | Session-based emotion scores, mood trends, and conversation memory |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│   login.html  │  mirror.html  │  style.css  │    app.js     │
├─────────────────────────────────────────────────────────────┤
│                    BUSINESS LOGIC LAYER                      │
│  app.py (Flask)  │  conversation.py  │  agentic_ai.py       │
│  emotion_detection.py  │  voice_sentiment.py                 │
│  explainable_ai.py  │  sacred_wisdom.py                      │
├─────────────────────────────────────────────────────────────┤
│                      DATA LAYER                              │
│  database.py (SQLite ORM)  │  emotions.db  │  Groq Cloud API │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3 (Glassmorphism), JavaScript (ES6+), Web Speech API |
| **Backend** | Python 3.10, Flask 3.x |
| **AI / ML** | DeepFace (facial emotion), Groq LLaMA 3.3 70B (chat), NLTK VADER + TextBlob (sentiment) |
| **Database** | SQLite3 |
| **Email** | Gmail SMTP (OTP password reset) |
| **Version Control** | Git + GitHub |

---

## 📁 Project Structure

```
echo_mirror/
├── app.py                  # Flask web server & API routes (838 lines)
├── conversation.py         # Multi-turn AI conversation manager (740 lines)
├── emotion_detection.py    # DeepFace facial emotion detection (744 lines)
├── database.py             # SQLite ORM — users, sessions, messages (666 lines)
├── agentic_ai.py           # Agentic AI engine — crisis detection, nudges (405 lines)
├── voice_sentiment.py      # Voice + text sentiment analysis (444 lines)
├── explainable_ai.py       # XAI — emotion explanation engine (311 lines)
├── sacred_wisdom.py        # Multi-faith wisdom quote database
├── visualize.py            # Emotion trend visualization (250 lines)
├── main.py                 # Standalone CLI version
├── requirements.txt        # Python dependencies
├── .env                    # API keys & secrets (not in repo)
├── .gitignore
├── static/
│   ├── app.js              # Frontend JavaScript (406 lines)
│   └── style.css           # Glassmorphic dark theme (645 lines)
└── templates/
    ├── login.html           # Auth page — login / signup / OTP reset (270 lines)
    └── mirror.html          # Main app — chat, camera, panels (123 lines)
```

**Total: ~5,842 lines of code**

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Webcam (optional — for facial emotion detection)
- Microphone (optional — for voice input)
- Gmail account with App Password (for OTP email)
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/sania5h6/Echo_mirror.git
cd Echo_mirror

# 2. Create virtual environment
python -m venv echomirror
# Windows:
echomirror\Scripts\activate
# Linux/Mac:
source echomirror/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (one-time)
python -c "import nltk; nltk.download('vader_lexicon')"
```

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
ECHO_DB_SECRET=your_random_secret_key
SMTP_EMAIL=your_gmail@gmail.com
SMTP_PASSWORD=your_gmail_app_password
```

> **Gmail App Password**: Go to [Google Account → Security → App Passwords](https://myaccount.google.com/apppasswords) to generate one.

### Run

```bash
python app.py
```

Open your browser and go to: **http://localhost:5000**

> ⚠️ Use `localhost` (not IP address) for camera/mic to work without HTTPS.

---

## 📸 Screenshots

### Login / Signup Page
- Glassmorphic dark theme with gradient background
- Login, Signup, and Forgot Password (OTP) flows
- Password visibility toggle

### Mirror Page (Main App)
- **Left Panel**: Camera view, "Why This Emotion?" XAI, Emotion Score bars
- **Center**: Chat with AI — empathetic responses, wisdom quotes, recommendation cards
- **Right Panel**: Sacred Wisdom, Agentic AI recommendations, Session stats
- **Bottom**: Sentiment indicator, text input, mic button, send button

### Smart Recommendations
- 💪 **Motivation** — "Every storm runs out of rain..."
- 🛠️ **Coping Strategy** — "Try journaling: Write 3 things you're grateful for..."
- 😄 **Light Moment** — Contextual jokes to lighten the mood
- 💜 **Affirmation** — "I am worthy of love and kindness..."
- 🫁 **Breathing Exercise** — "Breathe in 4s → Hold 4s → Out 6s..."

---

## 🧪 How It Works

1. **User sends a message** (text or voice)
2. **Sentiment Analysis** — VADER + TextBlob + keyword boosting calculates polarity
3. **Facial Detection** (if camera on) — DeepFace analyzes webcam frame for 7 emotions
4. **AI Response** — Groq LLaMA 3.3 70B generates empathetic reply based on emotion + conversation history
5. **Smart Recommendation** — After understanding the situation (turn 4+), suggests coping strategies, motivation, humor, or breathing exercises based on emotional intensity
6. **Sacred Wisdom** — Contextual quotes from Quran, Bible, Bhagavad Gita, Buddha
7. **Crisis Detection** — If distress keywords detected, provides helpline numbers immediately

---

## 🔐 Security

- **Password Hashing**: bcrypt with ECHO_DB_SECRET pepper
- **Session Management**: Flask secure sessions
- **OTP Password Reset**: 6-digit code sent via Gmail SMTP, expires in 5 minutes
- **Input Validation**: Email format, password length, empty message checks
- **User Isolation**: Per-user session state, no data mixing between accounts

---

## 👥 Team

| Member | Role | Modules |
|--------|------|---------|
| **R. Manoj Naik** | Emotion Detection & UI | `emotion_detection.py`, Frontend (HTML/CSS/JS), `visualize.py` |
| **R. Indu** | Backend & Database | `app.py`, `conversation.py`, `database.py` |
| **SK. Sania** |Agentic AI & NLP | `agentic_ai.py`, `voice_sentiment.py`, `explainable_ai.py`, `sacred_wisdom.py` |
---

## 🔮 Future Scope

- 🌐 **Cloud Deployment** — Deploy on AWS/GCP with Gunicorn + Nginx
- 🗣️ **Advanced TTS** — ElevenLabs or Google Cloud TTS for natural voice responses
- 📊 **Analytics Dashboard** — Long-term mood trends, emotion heatmaps
- 📱 **Mobile App** — React Native / Flutter version
- 🧪 **Automated Testing** — pytest + Selenium test suite
- 🔄 **Real-time WebSocket** — Replace polling with WebSocket for instant updates

---

## 📄 License

This project is developed as part of the B.Tech Final Year Project at GCET (Geethanjali College of Engineering and Technology).

---

<p align="center">
  <strong>🪞 EchoMirror — Your Reflection, Your Healing</strong><br>
  <em>Team Echo Mirror — R. Manoj Naik | R. Indu | SK. Sania</em>
</p>
