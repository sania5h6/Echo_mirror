"""
EchoMirror - Step 3: Conversational AI Module (v4)
Updates:
  - Shorter responses (2-3 sentences max, strict)
  - EchoMirror SPEAKS back using pyttsx3 TTS
  - Fixed sacred wisdom triggers (more frequent)
  - Fixed breathing exercise triggers
  - Voice input + Text input both supported
"""

import os
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
import tkinter as tk
from tkinter import scrolledtext
import threading
import queue
import random
import whisper
import torch
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import time
from gtts import gTTS
import pygame
import io
from agentic_ai import AgenticAI

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
GROQ_MODEL        = "llama-3.3-70b-versatile"
MAX_TOKENS        = 150      # Slightly more room for natural responses
TEMPERATURE       = 0.75     # Slightly lower = more focused, less random
MAX_HISTORY       = 10
SAMPLE_RATE       = 16000
CHUNK_SECONDS     = 3
SILENCE_THRESHOLD = 0.0005  # Lowered for better mic sensitivity
MIC_DEVICE        = 1      # 1=Intel built-in, 2=Boult headset, 15=Boult WASAPI
MIC_GAIN          = 5.0    # Boost mic sensitivity (increase if too quiet)
WHISPER_MODEL     = "medium"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
TTS_ACCENT        = "co.in"  # Indian English accent
# Options: co.in=Indian, com=American, co.uk=British, com.au=Australian

# Colors
BG         = "#0a0a14"   # Deeper dark — feels more like a mirror
CARD_BG    = "#12122a"   # Slightly blue tinted dark
ACCENT     = "#7c6ef7"   # Softer purple
TEXT_WHITE = "#eaf0fb"   # Warm white
TEXT_GRAY  = "#5a6477"   # Muted gray
TEXT_GREEN = "#43d49c"   # Teal green
TEXT_GOLD  = "#f0c060"   # Warm gold
USER_CLR   = "#60aaff"   # Clear blue
BOT_CLR    = "#b09cff"   # Soft lavender
WISDOM_CLR = "#ffe599"   # Warm yellow
SYSTEM_CLR = "#888888"   # Neutral system messages
MIC_ON     = "#ff6b6b"   # Red when recording
MIC_OFF    = "#252535"   # Dark when off


# ─────────────────────────────────────────
# TTS ENGINE — gTTS (Google TTS, Indian English)
# ─────────────────────────────────────────
class TTSEngine:
    def __init__(self):
        self._queue = queue.Queue()
        pygame.mixer.init(frequency=44100)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[TTS] gTTS engine ready (Indian English)")

    def _run_loop(self):
        while True:
            try:
                text = self._queue.get()
                if text is None or text == "__stop__":
                    continue
                if not text.strip():
                    continue
                tts = gTTS(text=text, lang="en", tld="co.in", slow=False)
                mp3_fp = io.BytesIO()
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                pygame.mixer.music.load(mp3_fp)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
                print(f"[TTS] Done speaking.")
            except Exception as e:
                print(f"[TTS Error] {e}")

    def speak(self, text: str):
        while not self._queue.empty():
            try: self._queue.get_nowait()
            except: pass
        self._queue.put(text)
        print(f"[TTS] Speaking: {text[:50]}...")

    def stop(self):
        pygame.mixer.music.stop()
        while not self._queue.empty():
            try: self._queue.get_nowait()
            except: pass

# ─────────────────────────────────────────
# SACRED QUOTES
# ─────────────────────────────────────────
SACRED_QUOTES = {
    "sad": [
        {"text": "Verily, with hardship comes ease.", "source": "Quran 94:5"},
        {"text": "The Lord is close to the brokenhearted and saves those who are crushed in spirit.", "source": "Bible, Psalm 34:18"},
        {"text": "Even the darkest night will end and the sun will rise.", "source": "Buddha Dhammapada"},
        {"text": "The impediment to action advances action. What stands in the way becomes the way.", "source": "Marcus Aurelius"},
    ],
    "angry": [
        {"text": "The strong person controls himself when angry.", "source": "Prophet Muhammad (PBUH)"},
        {"text": "A gentle answer turns away wrath.", "source": "Bible, Proverbs 15:1"},
        {"text": "Holding on to anger is like grasping a hot coal with the intent of throwing it at someone else.", "source": "Buddha"},
    ],
    "fear": [
        {"text": "Allah does not burden a soul beyond that it can bear.", "source": "Quran 2:286"},
        {"text": "Do not be afraid, for I am with you.", "source": "Bible, Isaiah 41:10"},
        {"text": "Fear not. What is not real never was and never will be.", "source": "Bhagavad Gita 2:16"},
        {"text": "You have power over your mind, not outside events.", "source": "Marcus Aurelius"},
    ],
    "neutral": [
        {"text": "Allah will not change the condition of a people until they change what is in themselves.", "source": "Quran 13:11"},
        {"text": "Be still and know that I am God.", "source": "Bible, Psalm 46:10"},
        {"text": "Peace comes from within. Do not seek it without.", "source": "Buddha"},
    ],
    "happy": [
        {"text": "And He found you lost and guided you.", "source": "Quran 93:7"},
        {"text": "This is the day that the Lord has made; let us rejoice and be glad in it.", "source": "Bible, Psalm 118:24"},
        {"text": "Happiness never decreases by being shared.", "source": "Buddha"},
    ],
    "disgust": [
        {"text": "Speak good words or remain silent.", "source": "Prophet Muhammad (PBUH)"},
        {"text": "If it is not right, do not do it; if it is not true, do not say it.", "source": "Marcus Aurelius"},
    ],
    "surprise": [
        {"text": "For I know the plans I have for you, plans to prosper you.", "source": "Bible, Jeremiah 29:11"},
        {"text": "The mind is everything. What you think you become.", "source": "Buddha"},
        {"text": "Life is what happens when you are busy making other plans.", "source": "Stoic wisdom"},
    ],
    "anxiety": [
        {"text": "Sufficient for you is Allah as a disposer of affairs.", "source": "Quran 4:81"},
        {"text": "Do not worry about tomorrow, for tomorrow will worry about itself.", "source": "Bible, Matthew 6:34"},
        {"text": "You suffer more in imagination than in reality.", "source": "Seneca"},
        {"text": "Wherever you go, go with all your heart.", "source": "Confucius"},
    ],
    "lonely": [
        {"text": "We are not alone when we think we are alone.", "source": "Rumi"},
        {"text": "The Lord himself goes before you and will be with you.", "source": "Bible, Deuteronomy 31:8"},
        {"text": "Loneliness adds beauty to life. It puts a special burn on sunsets.", "source": "Henry Rollins"},
    ],
    "depression": [
        {"text": "With every hardship comes ease — this too shall pass.", "source": "Quran 94:5-6"},
        {"text": "Come to me, all you who are weary and burdened, and I will give you rest.", "source": "Bible, Matthew 11:28"},
        {"text": "You deserve your own love and affection.", "source": "Buddha"},
        {"text": "After every difficulty, Allah has promised ease. Hold on.", "source": "Quran, inspired"},
    ],
    "lost": [
        {"text": "Whoever relies upon Allah, then He is sufficient for him.", "source": "Quran 65:3"},
        {"text": "Trust in the Lord with all your heart.", "source": "Bible, Proverbs 3:5"},
        {"text": "You are what your deep, driving desire is.", "source": "Brihadaranyaka Upanishad"},
        {"text": "It is not death that a man should fear, but never beginning to live.", "source": "Marcus Aurelius"},
    ],
}

COPING_EXERCISES = {
    "angry": [
        "Try box breathing: Inhale 4 counts, hold 4, exhale 4, hold 4. Repeat 3 times.",
        "Name 5 things you can see right now. This grounds you in the present.",
    ],
    "fear": [
        "Try 4-7-8 breathing: Inhale 4 counts, hold 7, exhale 8. It calms your nervous system.",
        "Place both feet flat on the floor. Feel the ground. You are safe right now.",
    ],
    "sad": [
        "Put one hand on your heart. Take three slow breaths. You are allowed to feel this.",
        "Write three things, however small, that you are grateful for today.",
    ],
    "neutral": [
        "Close your eyes, breathe slowly, and notice how your body feels right now.",
    ],
}


# ─────────────────────────────────────────
# WISDOM HELPERS — FIXED TRIGGERS
# ─────────────────────────────────────────
HEAVY_KEYWORDS = [
    "sad", "depressed", "lost", "hopeless", "scared", "angry", "frustrated",
    "empty", "alone", "hurt", "pain", "cry", "give up", "no point",
    "worthless", "broken", "tired", "meaningless", "exhausted", "pointless"
]

def get_relevant_quote(emotion, message):
    msg = message.lower()
    if any(w in msg for w in ["hopeless", "empty", "worthless", "give up", "no point", "meaningless", "no reason"]):
        pool = SACRED_QUOTES["depression"]
    elif any(w in msg for w in ["lost", "purpose", "meaning", "direction", "what am i", "who am i", "career", "future"]):
        pool = SACRED_QUOTES["lost"]
    elif any(w in msg for w in ["anxious", "anxiety", "worry", "worried", "overthink", "panic"]):
        pool = SACRED_QUOTES.get("anxiety", SACRED_QUOTES["fear"])
    elif any(w in msg for w in ["alone", "lonely", "no one", "nobody", "isolated"]):
        pool = SACRED_QUOTES.get("lonely", SACRED_QUOTES["sad"])
    else:
        pool = SACRED_QUOTES.get(emotion, SACRED_QUOTES["neutral"])
    return random.choice(pool)


def should_offer_wisdom(message, emotion, turn):
    """Offer wisdom when emotionally relevant — not mechanically every 2 turns"""
    msg = message.lower()
    is_heavy = any(w in msg for w in HEAVY_KEYWORDS)
    is_negative = emotion in ["sad", "angry", "fear", "disgust"]
    # Offer on turn 3, 6, 9... but only when emotionally relevant
    # Also offer on turn 2 if VERY heavy message
    if turn <= 3:
        return False  # Build connection first — no quotes early
    if is_heavy and turn == 4:
        return True  # First quote opportunity at turn 4
    return (is_heavy or is_negative) and turn % 5 == 0  # Then every 5 turns


def should_suggest_exercise(emotion, turn):
    """Only offer exercise after turn 4 — not too early"""
    return emotion in ["angry", "fear", "sad"] and turn >= 4 and turn % 4 == 0


# ─────────────────────────────────────────
# SYSTEM PROMPT — STRICT SHORT RESPONSES
# ─────────────────────────────────────────
def build_system_prompt(emotion, sentiment, polarity, memory_context=""):
    emotion_guidance = {
        "happy":    "They look happy — match their warmth, celebrate what's good.",
        "sad":      "They look sad — be slower, softer. Don't rush to fix. Just witness.",
        "angry":    "They look angry — stay grounded. Name the frustration before anything else.",
        "fear":     "They look fearful — be steady and reassuring. Keep your tone calm.",
        "surprise": "They look surprised — be curious, open, engaged.",
        "disgust":  "They look uncomfortable — validate without judgment.",
        "neutral":  "They look calm — be warm and gently curious.",
    }.get(emotion, "Be warm and fully present.")

    polarity_note = ""
    if polarity < -0.5:
        polarity_note = "⚠️ Deep negative emotion in words. Go very slow. One sentence at a time."
    elif polarity < -0.2:
        polarity_note = "Mild negative emotion. Be gentle and curious."
    elif polarity > 0.4:
        polarity_note = "Positive energy in their words. Affirm and celebrate with them."

    return f"""You are EchoMirror. You live in a mirror. You speak like a real friend — not a therapist.

USER: face={emotion}, sentiment={polarity:+.2f}
{emotion_guidance}
{polarity_note}
{memory_context}

RULES (follow exactly):
1. Max 2 sentences. HARD limit.
2. Questions must be SHORT and SIMPLE. Like a friend asking, not an interviewer.
3. Never be heavy or philosophical unless the user goes there first.
4. If user says something vague ("idk", "maybe", "nothing") — be gentle, don't push hard.
5. Speak casually. Contractions. Simple words. No fancy metaphors.
6. You know Indian student life — parental pressure, career confusion, "log kya kahenge".

BANNED WORDS/PHRASES (never use any of these):
- "can be", "may be", "could be" (too vague)
- "overwhelming", "crushing", "devastating" (too dramatic)  
- "I understand", "I'm sorry", "That must be"
- "It's important to", "I'm here for you", "You're not alone"
- "comparison can be tough", "loneliness can be"
- Any question longer than 12 words

GOOD RESPONSE STYLE:
User: "I feel like I'm disappointing everyone"
BAD:  "That weight of expectation can be crushing. What's the one thing you feel you're not living up to that's bothering you the most?"
GOOD: "Who specifically are you worried about disappointing?"

User: "my friends don't understand me"
BAD:  "It can feel really lonely when it seems like they don't get you. Are they saying something specific that's making you feel this way?"
GOOD: "What do you wish they understood?"

User: "i'm scared about my future"
BAD:  "The unknown can be scary, especially with everyone's expectations around you — what's one thing you secretly wish you could do?"
GOOD: "What's the scariest part — the not knowing, or what people will think?"

User: "i just feel empty"
BAD:  "That emptiness can be really hard to shake — is it like you're just going through the motions?"
GOOD: "Like nothing actually lands. When did it start feeling like that?"

User: "everyone else seems to have it together"
BAD:  "Comparison can be tough — do you think anyone really has it all figured out, or is it just a facade?"
GOOD: "Social media lied to all of us. What does 'having it together' even mean to you?"

User: "idk maybe"
BAD:  "The uncertainty can be overwhelming. What's on your mind?"
GOOD: "Take your time. What's the thing you keep pushing to the back of your head?"

User: "nothing" or one word answers
GOOD: "That's okay. I'm not going anywhere."

REMEMBER: Short. Simple. Warm. Human. Never robotic.
"""


# ─────────────────────────────────────────
# VOICE RECORDER
# ─────────────────────────────────────────
class VoiceRecorder:
    def __init__(self, whisper_model):
        self.model = whisper_model
        self.is_recording = False
        self.transcript = ""

    def record_chunk(self):
        # Device 2 = Boult headset mic, Device 1 = Intel built-in mic
        # Using device 2 (headset) for better quality
        # Change MIC_DEVICE to 1 if not using headset
        chunk = sd.rec(int(CHUNK_SECONDS * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                       device=MIC_DEVICE)
        sd.wait()
        audio = chunk.flatten()
        # Boost audio gain x5 for low-sensitivity mics
        audio = audio * MIC_GAIN
        audio = np.clip(audio, -1.0, 1.0)
        return audio

    # Whisper hallucination phrases — background noise triggers these
    HALLUCINATIONS = {
        # Common Whisper hallucinations on silence/noise
        "thank you", "thanks", "thank you.", "thanks.", "thank you so much",
        "you", ".", "", " ", "bye", "bye.", "goodbye",
        "okay", "ok", "okay.", "ok.", "alright",
        "hmm", "hmm.", "um", "uh", "uh.", "ah", "oh",
        "subscribe", "like and subscribe", "please subscribe",
        "ready set go", "ready, set, go",
        "this is a test", "let's test this", "testing testing",
        "let's test this setting", "one two three",
        "for more videos", "see you next time",
        "watch till the end", "have a nice day",
    }

    def transcribe_chunk(self, audio):
        avg = np.abs(audio).mean()
        print(f"[MIC] Audio level: {avg:.5f} (threshold: {SILENCE_THRESHOLD})")
        if avg < SILENCE_THRESHOLD:
            print("[MIC] Silent chunk, skipping...")
            return ""
        # Extra check: if avg is very low even above threshold, likely noise
        if avg < 0.003:
            print("[MIC] Low energy, likely background noise, skipping...")
            return ""
        print("[MIC] Sound detected! Transcribing...")
        tmp_path = "_mic_chunk.wav"
        try:
            audio_int = (audio * 32767).astype(np.int16)
            wav.write(tmp_path, SAMPLE_RATE, audio_int)
            result = self.model.transcribe(
                tmp_path,
                fp16=(DEVICE == "cuda"),
                language="en",           # Force English — prevents Japanese hallucinations
                no_speech_threshold=0.6, # Whisper's own silence detector
                logprob_threshold=-1.0,  # Filter low-confidence words
                condition_on_previous_text=False,  # Prevents looping same phrase
                temperature=0.0,         # Greedy decoding — most accurate
            )
            text = result["text"].strip()

            # Filter hallucinations
            if text.lower().strip("., ") in self.HALLUCINATIONS:
                print(f"[MIC] Hallucination filtered: {text}")
                return ""

            # Filter very short responses (likely noise)
            if len(text.split()) < 2:
                print(f"[MIC] Too short, likely noise: {text}")
                return ""

            print(f"[MIC] Transcribed: {text}")
            return text
        except Exception as e:
            print(f"[Whisper Error] {e}")
            return ""
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except:
                pass

    def start(self, on_update):
        self.is_recording = True
        self.transcript = ""
        while self.is_recording:
            chunk = self.record_chunk()
            text = self.transcribe_chunk(chunk)
            if text:
                self.transcript = (self.transcript + " " + text).strip()
                on_update(self.transcript)

    def stop(self):
        self.is_recording = False
        time.sleep(0.3)
        return self.transcript


# ─────────────────────────────────────────
# CONVERSATION ENGINE
# ─────────────────────────────────────────
class ConversationEngine:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file!")
        self.client = Groq(api_key=api_key)
        self.history = []
        self.emotion = "neutral"
        self.sentiment = "Neutral"
        self.polarity = 0.0
        self.turn = 0
        self.memory_context = ""

    def update_emotional_state(self, emotion, sentiment, polarity):
        self.emotion = emotion
        self.sentiment = sentiment
        self.polarity = polarity

    def set_memory_context(self, context):
        self.memory_context = context

    def chat(self, user_message):
        self.turn += 1
        wisdom_used = None
        exercise_used = None
        extra_context = ""

        if should_offer_wisdom(user_message, self.emotion, self.turn):
            quote = get_relevant_quote(self.emotion, user_message)
            wisdom_used = quote
            extra_context += f'\n\nNaturally include this wisdom in ONE sentence:\n"{quote["text"]}" — {quote["source"]}'

        if should_suggest_exercise(self.emotion, self.turn):
            exercises = COPING_EXERCISES.get(self.emotion, COPING_EXERCISES["neutral"])
            exercise_used = random.choice(exercises)
            extra_context += f"\n\nSuggest this in ONE gentle sentence:\n{exercise_used}"

        system_prompt = build_system_prompt(
            self.emotion, self.sentiment, self.polarity, self.memory_context
        ) + extra_context

        self.history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": system_prompt}] + self.history[-MAX_HISTORY:]

        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL, messages=messages,
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            )
            reply = response.choices[0].message.content.strip()
            self.history.append({"role": "assistant", "content": reply})
            self._log(user_message, reply, wisdom_used)
            return reply, wisdom_used, exercise_used
        except Exception as e:
            print(f"[Engine Error] {e}")
            return "Something went wrong. Please try again.", None, None

    def _log(self, user_msg, reply, wisdom):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{ts}] Turn {self.turn} | {self.emotion} | {self.sentiment}")
        print(f"  User : {user_msg[:80]}")
        print(f"  Echo : {reply}")
        if wisdom:
            print(f"  Quote: {wisdom['text'][:60]} — {wisdom['source']}")

    def reset_session(self):
        self.history = []
        self.turn = 0


# ─────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────
class EchoMirrorChatUI:
    def __init__(self, engine, whisper_model):
        self.engine = engine
        self.recorder = VoiceRecorder(whisper_model)
        self.tts = TTSEngine()
        self.agent = AgenticAI()
        self.ui_queue = queue.Queue()
        self.is_recording = False
        self.tts_enabled = True

        self.root = tk.Tk()
        self.root.title("EchoMirror — Conversation")
        self.root.configure(bg=BG)
        self.root.geometry("860x800")
        self.root.resizable(False, False)

        self._build_ui()
        self.root.after(100, self._process_queue)
        self.root.after(600, self._send_welcome)

    def _build_ui(self):
        # Header
        tk.Label(self.root, text="EchoMirror", font=("Segoe UI", 24, "bold"),
                 bg=BG, fg=ACCENT).pack(pady=(16, 2))
        tk.Label(self.root, text="The power to heal lies within you",
                 font=("Segoe UI", 10, "italic"), bg=BG, fg=TEXT_GRAY).pack(pady=(0,4))

        # State bar
        state_frame = tk.Frame(self.root, bg=CARD_BG, pady=6)
        state_frame.pack(fill=tk.X, padx=20, pady=(8, 4))
        tk.Label(state_frame, text="State:", font=("Arial", 9, "bold"),
                 bg=CARD_BG, fg=TEXT_GRAY).pack(side=tk.LEFT, padx=10)
        self.emotion_label = tk.Label(state_frame, text="Face: neutral",
                                      font=("Arial", 9), bg=CARD_BG, fg=TEXT_GOLD)
        self.emotion_label.pack(side=tk.LEFT, padx=6)
        self.sentiment_label = tk.Label(state_frame, text="Sentiment: Neutral",
                                        font=("Arial", 9), bg=CARD_BG, fg=TEXT_GREEN)
        self.sentiment_label.pack(side=tk.LEFT, padx=6)

        # TTS toggle
        self.tts_var = tk.BooleanVar(value=True)
        tts_check = tk.Checkbutton(
            state_frame, text="🔊 Voice", variable=self.tts_var,
            bg=CARD_BG, fg=TEXT_GREEN, selectcolor=CARD_BG,
            activebackground=CARD_BG, font=("Arial", 9),
            command=self._toggle_tts
        )
        tts_check.pack(side=tk.RIGHT, padx=10)

        # Override dropdown
        tk.Label(state_frame, text="Override:", font=("Arial", 9),
                 bg=CARD_BG, fg=TEXT_GRAY).pack(side=tk.LEFT, padx=(16, 4))
        self.emo_var = tk.StringVar(value="neutral")
        emo_menu = tk.OptionMenu(state_frame, self.emo_var,
                                  *["happy","sad","angry","fear","surprise","neutral","disgust"],
                                  command=self._update_emotion)
        emo_menu.config(bg=CARD_BG, fg="white", font=("Arial", 8),
                        highlightthickness=0, relief=tk.FLAT, activebackground=ACCENT)
        emo_menu["menu"].config(bg=CARD_BG, fg="white")
        emo_menu.pack(side=tk.LEFT)

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            self.root, width=88, height=22,
            font=("Segoe UI", 11), bg=CARD_BG, fg=TEXT_WHITE,
            insertbackground="white", relief=tk.FLAT,
            wrap=tk.WORD, state=tk.DISABLED
        )
        self.chat_display.pack(padx=20, pady=(4, 4))
        self.chat_display.tag_config("user",   foreground=USER_CLR)
        self.chat_display.tag_config("echo",   foreground=BOT_CLR)
        self.chat_display.tag_config("meta",   foreground=TEXT_GRAY)
        self.chat_display.tag_config("wisdom", foreground=WISDOM_CLR,
                                     font=("Arial", 10, "italic"))
        self.chat_display.tag_config("system", foreground=SYSTEM_CLR, font=("Segoe UI", 9, "italic"))
        self.chat_display.tag_config("user",   foreground=USER_CLR, font=("Segoe UI", 11))
        self.chat_display.tag_config("echo",   foreground=BOT_CLR,  font=("Segoe UI", 11))
        self.chat_display.tag_config("meta",   foreground=TEXT_GRAY,font=("Segoe UI", 9))
        self.chat_display.tag_config("wisdom", foreground=WISDOM_CLR,font=("Segoe UI", 10, "italic"))

        # Input area
        input_outer = tk.Frame(self.root, bg=BG)
        input_outer.pack(fill=tk.X, padx=20, pady=4)

        self.input_box = tk.Text(
            input_outer, width=57, height=3,
            font=("Segoe UI", 11), bg=CARD_BG, fg=TEXT_WHITE,
            insertbackground="white", relief=tk.FLAT, wrap=tk.WORD
        )
        self.input_box.pack(side=tk.LEFT, padx=(0, 6))
        self.input_box.bind("<Return>", self._on_enter)

        # Send + Reset buttons
        btn_col = tk.Frame(input_outer, bg=BG)
        btn_col.pack(side=tk.LEFT, padx=(0, 6))
        self.send_btn = tk.Button(
            btn_col, text="SEND", width=9, height=1,
            font=("Arial", 11, "bold"), bg=ACCENT, fg="white",
            activebackground=BOT_CLR, relief=tk.FLAT, cursor="hand2",
            command=self.send_text_message
        )
        self.send_btn.pack(pady=(0, 4))
        self.reset_btn = tk.Button(
            btn_col, text="RESET", width=9, height=1,
            font=("Arial", 10), bg=CARD_BG, fg=TEXT_GRAY,
            activebackground="#2d3436", relief=tk.FLAT, cursor="hand2",
            command=self.reset_session
        )
        self.reset_btn.pack()

        # Mic + Done buttons
        mic_col = tk.Frame(input_outer, bg=BG)
        mic_col.pack(side=tk.LEFT)
        self.mic_btn = tk.Button(
            mic_col, text="🎙 SPEAK", width=10, height=1,
            font=("Arial", 11, "bold"), bg=MIC_OFF, fg="white",
            activebackground=MIC_ON, relief=tk.FLAT, cursor="hand2",
            command=self.start_voice
        )
        self.mic_btn.pack(pady=(0, 4))
        self.done_btn = tk.Button(
            mic_col, text="✓ DONE", width=10, height=1,
            font=("Arial", 11, "bold"), bg="#00b894", fg="white",
            activebackground="#55efc4", relief=tk.FLAT, cursor="hand2",
            state=tk.DISABLED, command=self.stop_voice
        )
        self.done_btn.pack()

        # Status
        self.status_label = tk.Label(
            self.root,
            text="Type a message and press ENTER  |  or click 🎙 SPEAK to talk",
            font=("Segoe UI", 9, "italic"), bg=BG, fg=TEXT_GRAY
        )
        self.status_label.pack(pady=(2, 8))

    # ─── TTS ───
    def _toggle_tts(self):
        self.tts_enabled = self.tts_var.get()
        state = "ON" if self.tts_enabled else "OFF"
        print(f"[TTS] Voice output: {state}")

    # ─── VOICE INPUT ───
    def start_voice(self):
        self.is_recording = True
        self.mic_btn.config(bg=MIC_ON, text="🔴 LIVE", state=tk.DISABLED)
        self.done_btn.config(state=tk.NORMAL)
        self.send_btn.config(state=tk.DISABLED)
        self.input_box.config(state=tk.DISABLED)
        self.tts.stop()  # Stop any ongoing speech when user starts talking
        self.status_label.config(text="🔴 Recording... Whisper is transcribing. Edit anytime. Click DONE when ready.")
        self._clear_input()
        self.input_box.config(state=tk.NORMAL)  # Keep editable during recording

        def on_update(text):
            self.ui_queue.put(("live_transcript", text))

        threading.Thread(target=self.recorder.start,
                         args=(on_update,), daemon=True).start()

    def stop_voice(self):
        self.is_recording = False
        self.mic_btn.config(bg=MIC_OFF, text="🎙 SPEAK", state=tk.NORMAL)
        self.done_btn.config(state=tk.DISABLED)
        self.send_btn.config(state=tk.NORMAL)
        self.input_box.config(state=tk.NORMAL)
        self.status_label.config(text="Processing your voice...")

        def finish():
            text = self.recorder.stop()
            if text:
                self.ui_queue.put(("voice_done", text))
            else:
                self.ui_queue.put(("status", "No speech detected. Try again."))

        threading.Thread(target=finish, daemon=True).start()

    # ─── TEXT INPUT ───
    def _on_enter(self, event):
        if not event.state & 0x1:
            self.send_text_message()
            return "break"

    def send_text_message(self):
        text = self.input_box.get("1.0", tk.END).strip()
        if not text:
            return
        self._clear_input()
        self._dispatch_message(text)

    def _dispatch_message(self, text):
        self._append("You", text, "user")
        self.send_btn.config(state=tk.DISABLED)
        self.mic_btn.config(state=tk.DISABLED)
        self.status_label.config(text="EchoMirror is thinking...")
        threading.Thread(target=self._get_response, args=(text,), daemon=True).start()

    def _get_response(self, text):
        # Run agentic checks BEFORE getting AI reply
        agentic = self.agent.process_turn(
            text, self.engine.emotion, self.engine.polarity, self.engine.turn + 1
        )

        # Crisis — override everything
        if agentic["crisis"]:
            self.ui_queue.put(("reply", agentic["crisis_response"], None, None))
            return

        # Normal AI reply
        reply, wisdom, exercise = self.engine.chat(text)

        # Append agentic extras to reply
        extras = []
        if agentic["breathing_exercise"]:
            extras.append("" + agentic["breathing_exercise"])
        if agentic["goal_check"]:
            extras.append("💭 " + agentic["goal_check"])
        if agentic["motivational_nudge"]:
            extras.append("" + agentic["motivational_nudge"])

        if extras:
            reply = reply + "".join(extras)

        self.ui_queue.put(("reply", reply, wisdom, exercise))

    # ─── SESSION ───
    def reset_session(self):
        self.tts.stop()
        self.engine.reset_session()
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self._append("System", "Session reset. Starting fresh.", "system")
        self.root.after(500, self._send_welcome)

    def _send_welcome(self):
        # Use agentic opening based on history
        try:
            from database import MemoryDB
            db = MemoryDB()
            total = db.get_total_sessions()
            sad_streak = db.get_emotion_streak("sad")
            happy_streak = db.get_emotion_streak("happy")
            welcome = self.agent.get_opening_message(total, sad_streak, happy_streak)
        except Exception:
            welcome = "Hey, I'm EchoMirror. This is your space — no judgment, no rush. How are you feeling right now?"
        self._append("EchoMirror", welcome, "echo")
        if self.tts_enabled:
            self.tts.speak(welcome)

    def _update_emotion(self, value):
        self.engine.update_emotional_state(value, self.engine.sentiment, self.engine.polarity)
        self.emotion_label.config(text=f"Face: {value}")

    # ─── UI HELPERS ───
    def _clear_input(self):
        self.input_box.config(state=tk.NORMAL)
        self.input_box.delete("1.0", tk.END)

    def _append(self, sender, message, tag, wisdom=None):
        self.chat_display.config(state=tk.NORMAL)
        ts = datetime.now().strftime("%H:%M")
        if tag == "user":
            turn_info = f"  Turn {self.engine.turn}" if self.engine.turn > 0 else ""
            self.chat_display.insert(tk.END, f"\n[{ts}] You{turn_info}\n", "meta")
            self.chat_display.insert(tk.END, f"{message}\n", "user")
        elif tag == "echo":
            emotion = self.engine.emotion.upper()
            self.chat_display.insert(tk.END, f"\n[{ts}] EchoMirror  [{emotion}]\n", "meta")
            self.chat_display.insert(tk.END, f"{message}\n", "echo")
            if wisdom:
                self.chat_display.insert(tk.END, f'\n  ✨ {wisdom["text"]}\n', "wisdom")
        elif tag == "system":
            self.chat_display.insert(tk.END, f"\n— {message} —\n", "system")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def _set_live_transcript(self, text):
        # Always keep editable — user can fix Whisper errors while speaking
        self.input_box.config(state=tk.NORMAL)
        self.input_box.delete("1.0", tk.END)
        self.input_box.insert(tk.END, text)
        # Cursor to end so user can keep typing
        self.input_box.mark_set(tk.INSERT, tk.END)

    # ─── QUEUE ───
    def _process_queue(self):
        try:
            while True:
                item = self.ui_queue.get_nowait()

                if item[0] == "live_transcript":
                    self._set_live_transcript(item[1])

                elif item[0] == "voice_done":
                    # Put transcribed text in editable box — let user review/edit before sending
                    self.input_box.config(state=tk.NORMAL)
                    self.input_box.delete("1.0", tk.END)
                    self.input_box.insert(tk.END, item[1])
                    self.input_box.focus_set()
                    self.send_btn.config(state=tk.NORMAL)
                    self.mic_btn.config(state=tk.NORMAL)
                    self.status_label.config(
                        text="✏️ Check transcript below — edit if needed, then press SEND")

                elif item[0] == "reply":
                    _, reply, wisdom, exercise = item
                    self._append("EchoMirror", reply, "echo", wisdom)
                    # Speak the reply (TTS)
                    if self.tts_enabled:
                        speak_text = reply
                        if wisdom:
                            speak_text += f". {wisdom['text']}"
                        self.tts.speak(speak_text)
                    self.send_btn.config(state=tk.NORMAL)
                    self.mic_btn.config(state=tk.NORMAL)
                    self.status_label.config(
                        text="Type and press ENTER  |  or click 🎙 SPEAK to use your voice")
                    self.emotion_label.config(text=f"Face: {self.engine.emotion}")
                    self.sentiment_label.config(text=f"Sentiment: {self.engine.sentiment}")

                elif item[0] == "status":
                    self.status_label.config(text=item[1])
                    self.send_btn.config(state=tk.NORMAL)
                    self.mic_btn.config(state=tk.NORMAL)

        except queue.Empty:
            pass
        self.root.after(100, self._process_queue)

    def run(self):
        self.root.mainloop()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run():
    print("[EchoMirror] Conversation v4 — Voice + Text + TTS + Sacred Wisdom")
    print(f"Whisper device : {DEVICE.upper()}")
    print("Loading Whisper model...")
    whisper_model = whisper.load_model(WHISPER_MODEL, device=DEVICE)
    print("Whisper ready.")
    engine = ConversationEngine()
    app = EchoMirrorChatUI(engine, whisper_model)
    app.run()


if __name__ == "__main__":
    run()