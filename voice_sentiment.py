"""
EchoMirror - Step 2: Voice + Text Input & Sentiment Analysis (v4)
Changes:
  - User can choose: VOICE mode or TEXT mode
  - Voice mode: real-time transcription + Done button
  - Text mode: user types freely, clicks Analyse
  - Both modes give same sentiment analysis + EchoMirror response
  - GPU accelerated Whisper medium
  - VADER + TextBlob blended sentiment (v4)
  - Complete 35-entry emotion×sentiment fusion map
  - SentimentAnalyzer class for integration
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import threading
import tkinter as tk
from tkinter import scrolledtext, font as tkfont
import queue
import time
from textblob import TextBlob
from datetime import datetime

# VADER for better emotional text scoring
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    print("[VoiceSentiment] VADER not available — using TextBlob only")


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
SAMPLE_RATE = 16000
WHISPER_MODEL = "medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SECONDS = 3
SILENCE_THRESHOLD = 0.008

# Colors
BG         = "#0f0f1a"
CARD_BG    = "#1a1a2e"
ACCENT     = "#6c5ce7"
ACCENT2    = "#00b894"
TEXT_WHITE = "#dfe6e9"
TEXT_GRAY  = "#636e72"
TEXT_GREEN = "#55efc4"
TEXT_GOLD  = "#fdcb6e"
VOICE_CLR  = "#e17055"
TEXT_CLR   = "#74b9ff"


# ─────────────────────────────────────────
# SENTIMENT ANALYSIS
# ─────────────────────────────────────────
def get_sentiment_label(polarity):
    if polarity >= 0.5:
        return ("Very Positive", "You sound really uplifted! Keep going.")
    elif polarity >= 0.1:
        return ("Positive", "There's a positive energy in your words.")
    elif polarity >= -0.1:
        return ("Neutral", "You seem calm and composed.")
    elif polarity >= -0.5:
        return ("Negative", "It sounds like something's weighing on you.")
    else:
        return ("Very Negative", "You seem to be going through a tough time. You're not alone.")


def analyze_sentiment(text):
    """Blended sentiment: VADER (60%) + TextBlob (40%) for better emotional text scoring."""
    blob = TextBlob(text)
    tb_polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if HAS_VADER:
        vader_scores = _vader.polarity_scores(text)
        vader_polarity = vader_scores["compound"]  # -1 to +1
        # Weighted blend: VADER is better for emotional language
        polarity = 0.6 * vader_polarity + 0.4 * tb_polarity
    else:
        polarity = tb_polarity

    label, advice = get_sentiment_label(polarity)
    return {
        "text": text,
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3),
        "label": label,
        "advice": advice,
    }


def fuse_signals(face_emotion, voice_sentiment):
    """Complete 35-entry fusion map: 7 emotions × 5 sentiment levels."""
    fusion_map = {
        # Happy
        ("happy",    "Very Positive"): "You're absolutely glowing today! Love to see it.",
        ("happy",    "Positive"):      "You're genuinely joyful — keep it up!",
        ("happy",    "Neutral"):       "You seem at ease. That's a good place to be.",
        ("happy",    "Negative"):      "Your face smiles but your words suggest stress. Want to talk?",
        ("happy",    "Very Negative"): "You're putting on a brave face. It's okay to let the mask down here.",
        # Sad
        ("sad",      "Very Positive"): "You're finding light even in sadness — that's real strength.",
        ("sad",      "Positive"):      "You're staying strong despite the sadness — that's courage.",
        ("sad",      "Neutral"):       "You're holding it together quietly. That takes strength.",
        ("sad",      "Negative"):      "You seem really down. It's okay to feel this way.",
        ("sad",      "Very Negative"): "I can feel the weight you're carrying. Please be gentle with yourself.",
        # Angry
        ("angry",    "Very Positive"): "That fire in you is being channeled into something good.",
        ("angry",    "Positive"):      "You're frustrated but optimistic. That's powerful energy.",
        ("angry",    "Neutral"):       "Something's bothering you beneath the surface. I'm here.",
        ("angry",    "Negative"):      "You sound and look frustrated. Let's take a breath together.",
        ("angry",    "Very Negative"): "You're carrying a lot of anger right now. Let's find a release.",
        # Fear
        ("fear",     "Very Positive"): "You feel nervous but excited — that's butterflies, not fear!",
        ("fear",     "Positive"):      "You're anxious but hopeful. That courage will carry you.",
        ("fear",     "Neutral"):       "It's okay to feel uncertain. Take it one step at a time.",
        ("fear",     "Negative"):      "You seem anxious. Remember — you are safe.",
        ("fear",     "Very Negative"): "Fear and worry are heavy right now. Breathe — you're not alone.",
        # Surprise
        ("surprise", "Very Positive"): "Something wonderful just happened! Tell me everything!",
        ("surprise", "Positive"):      "Something exciting just happened! Tell me more!",
        ("surprise", "Neutral"):       "Something unexpected caught your attention. What was it?",
        ("surprise", "Negative"):      "That caught you off guard in a tough way. How are you processing it?",
        ("surprise", "Very Negative"): "That was a shock. Take a moment — I'm right here.",
        # Disgust
        ("disgust",  "Very Positive"): "Something bothers you, but you're focusing on the good. Respect.",
        ("disgust",  "Positive"):      "You're frustrated with something but handling it well.",
        ("disgust",  "Neutral"):       "Something doesn't sit right with you. Your instincts matter.",
        ("disgust",  "Negative"):      "That clearly bothers you deeply. Your feelings are valid.",
        ("disgust",  "Very Negative"): "You're really upset by something. Let's talk through it.",
        # Neutral
        ("neutral",  "Very Positive"): "You're calm and full of positivity. Beautiful balance.",
        ("neutral",  "Positive"):      "You're content and at peace.",
        ("neutral",  "Neutral"):       "You're calm and balanced right now.",
        ("neutral",  "Negative"):      "Your words carry some weight today. Want to share more?",
        ("neutral",  "Very Negative"): "You look calm, but your words tell a different story. I'm listening.",
    }
    key = (face_emotion.lower(), voice_sentiment)
    return fusion_map.get(key, f"I hear you. Whatever you're feeling right now is valid.")


# ─────────────────────────────────────────
# SENTIMENT ANALYZER CLASS (for integration)
# ─────────────────────────────────────────
class SentimentAnalyzer:
    """Importable class for conversation.py and ct.py integration."""

    @staticmethod
    def analyze(text: str) -> dict:
        return analyze_sentiment(text)

    @staticmethod
    def fuse(face_emotion: str, voice_sentiment: str) -> str:
        return fuse_signals(face_emotion, voice_sentiment)

    @staticmethod
    def get_polarity(text: str) -> float:
        result = analyze_sentiment(text)
        return result["polarity"]


def log_result(result, mode):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] [{mode.upper()} MODE]")
    print(f"  Text       : {result['text'][:80]}{'...' if len(result['text']) > 80 else ''}")
    print(f"  Sentiment  : {result['label']}")
    print(f"  Polarity   : {result['polarity']:+.3f} | Subjectivity: {result['subjectivity']:.3f}")
    print(f"  Advice     : {result['advice']}")
    print(f"  EchoMirror : {result.get('fused', '')}")


# ─────────────────────────────────────────
# VOICE SESSION
# ─────────────────────────────────────────
class VoiceSession:
    def __init__(self, model, ui_queue):
        self.model = model
        self.ui_queue = ui_queue
        self.is_recording = False
        self.full_transcript = ""

    def record_chunk(self):
        chunk = sd.rec(int(CHUNK_SECONDS * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        return chunk.flatten()

    def transcribe_chunk(self, audio):
        if np.abs(audio).mean() < SILENCE_THRESHOLD:
            return ""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.write(tmp.name, SAMPLE_RATE, (audio * 32767).astype(np.int16))
        try:
            result = self.model.transcribe(tmp.name, fp16=(DEVICE == "cuda"))
            return result["text"].strip()
        except:
            return ""
        finally:
            os.unlink(tmp.name)

    def start_recording(self):
        self.is_recording = True
        self.full_transcript = ""
        self.ui_queue.put(("status", "Listening... Speak your thoughts freely"))

        while self.is_recording:
            chunk = self.record_chunk()
            text = self.transcribe_chunk(chunk)
            if text:
                self.full_transcript = (self.full_transcript + " " + text).strip()
                self.ui_queue.put(("transcript", self.full_transcript))

        self.ui_queue.put(("status", "Analysing your words..."))

    def stop_and_analyse(self, face_emotion):
        self.is_recording = False
        time.sleep(0.5)
        if not self.full_transcript:
            self.ui_queue.put(("result", {"error": "No speech detected. Please try again."}))
            return
        result = analyze_sentiment(self.full_transcript)
        result["fused"] = fuse_signals(face_emotion, result["label"])
        log_result(result, "voice")
        self.ui_queue.put(("result", result))


# ─────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────
class EchoMirrorUI:
    def __init__(self, model):
        self.model = model
        self.ui_queue = queue.Queue()
        self.session = None
        self.current_mode = None   # "voice" or "text"

        self.root = tk.Tk()
        self.root.title("EchoMirror — Express Yourself")
        self.root.configure(bg=BG)
        self.root.geometry("750x680")
        self.root.resizable(False, False)

        self._build_ui()
        self.root.after(100, self._process_queue)

    def _build_ui(self):
        # ── Header ──
        tk.Label(self.root, text="EchoMirror", font=("Arial", 24, "bold"),
                 bg=BG, fg=ACCENT).pack(pady=(18, 2))
        tk.Label(self.root, text="The power to heal lies within you",
                 font=("Arial", 10, "italic"), bg=BG, fg=TEXT_GRAY).pack()

        # ── Mode Selection ──
        tk.Label(self.root, text="How would you like to express yourself today?",
                 font=("Arial", 11), bg=BG, fg=TEXT_WHITE).pack(pady=(18, 8))

        mode_frame = tk.Frame(self.root, bg=BG)
        mode_frame.pack()

        self.voice_btn = tk.Button(
            mode_frame, text="🎙  SPEAK", width=16, height=2,
            font=("Arial", 12, "bold"), bg=VOICE_CLR, fg="white",
            activebackground="#d63031", relief=tk.FLAT, cursor="hand2",
            command=lambda: self._set_mode("voice")
        )
        self.voice_btn.pack(side=tk.LEFT, padx=12)

        self.text_btn = tk.Button(
            mode_frame, text="✏  TYPE", width=16, height=2,
            font=("Arial", 12, "bold"), bg=TEXT_CLR, fg="white",
            activebackground="#0984e3", relief=tk.FLAT, cursor="hand2",
            command=lambda: self._set_mode("text")
        )
        self.text_btn.pack(side=tk.LEFT, padx=12)

        # ── Face emotion selector ──
        emo_frame = tk.Frame(self.root, bg=BG)
        emo_frame.pack(pady=(10, 0))
        tk.Label(emo_frame, text="Face Emotion:", font=("Arial", 9),
                 bg=BG, fg=TEXT_GRAY).pack(side=tk.LEFT, padx=5)
        self.emo_var = tk.StringVar(value="neutral")
        emotions = ["happy", "sad", "angry", "fear", "surprise", "neutral", "disgust"]
        emo_menu = tk.OptionMenu(emo_frame, self.emo_var, *emotions)
        emo_menu.config(bg=CARD_BG, fg="white", font=("Arial", 9),
                        activebackground=ACCENT, highlightthickness=0, relief=tk.FLAT)
        emo_menu["menu"].config(bg=CARD_BG, fg="white")
        emo_menu.pack(side=tk.LEFT)

        # ── Status ──
        self.status_label = tk.Label(self.root, text="Choose SPEAK or TYPE above to begin",
                                     font=("Arial", 10, "italic"), bg=BG, fg=TEXT_GOLD)
        self.status_label.pack(pady=(12, 4))

        # ── Mode indicator ──
        self.mode_label = tk.Label(self.root, text="", font=("Arial", 10, "bold"),
                                   bg=BG, fg=TEXT_GRAY)
        self.mode_label.pack()

        # ── Input area (changes based on mode) ──
        input_label_frame = tk.Frame(self.root, bg=BG)
        input_label_frame.pack(anchor="w", padx=30, pady=(8, 2))
        self.input_label = tk.Label(input_label_frame, text="Your words:",
                                    font=("Arial", 10, "bold"), bg=BG, fg="#b2bec3")
        self.input_label.pack(side=tk.LEFT)

        self.input_box = scrolledtext.ScrolledText(
            self.root, width=74, height=7,
            font=("Arial", 11), bg=CARD_BG, fg=TEXT_WHITE,
            insertbackground="white", relief=tk.FLAT, wrap=tk.WORD
        )
        self.input_box.pack(padx=30, pady=2)
        self.input_box.config(state=tk.DISABLED)  # Enabled when mode is chosen

        # ── Action buttons ──
        self.action_frame = tk.Frame(self.root, bg=BG)
        self.action_frame.pack(pady=8)

        self.start_btn = tk.Button(
            self.action_frame, text="START RECORDING", width=18, height=2,
            font=("Arial", 11, "bold"), bg=VOICE_CLR, fg="white",
            activebackground="#d63031", relief=tk.FLAT, cursor="hand2",
            state=tk.DISABLED, command=self.start_recording
        )
        self.start_btn.pack(side=tk.LEFT, padx=8)

        self.done_btn = tk.Button(
            self.action_frame, text="DONE / ANALYSE", width=18, height=2,
            font=("Arial", 11, "bold"), bg=ACCENT2, fg="white",
            activebackground="#00cec9", relief=tk.FLAT, cursor="hand2",
            state=tk.DISABLED, command=self.done_action
        )
        self.done_btn.pack(side=tk.LEFT, padx=8)

        self.clear_btn = tk.Button(
            self.action_frame, text="CLEAR", width=8, height=2,
            font=("Arial", 11), bg=CARD_BG, fg=TEXT_GRAY,
            activebackground="#2d3436", relief=tk.FLAT, cursor="hand2",
            command=self.clear_all
        )
        self.clear_btn.pack(side=tk.LEFT, padx=8)

        # ── Result area ──
        tk.Label(self.root, text="EchoMirror Response:",
                 font=("Arial", 10, "bold"), bg=BG, fg="#b2bec3").pack(anchor="w", padx=30, pady=(8, 2))
        self.result_box = scrolledtext.ScrolledText(
            self.root, width=74, height=5,
            font=("Arial", 11), bg=CARD_BG, fg=TEXT_GREEN,
            insertbackground="white", relief=tk.FLAT, wrap=tk.WORD
        )
        self.result_box.pack(padx=30, pady=(0, 10))
        self.result_box.config(state=tk.DISABLED)

    # ─── MODE SELECTION ───
    def _set_mode(self, mode):
        self.current_mode = mode
        self.clear_all()

        if mode == "voice":
            self.mode_label.config(text="VOICE MODE — Press START, speak, then DONE",
                                   fg=VOICE_CLR)
            self.input_box.config(state=tk.DISABLED, bg="#1a1a2e")
            self.start_btn.config(state=tk.NORMAL, text="START RECORDING", bg=VOICE_CLR)
            self.done_btn.config(state=tk.DISABLED, text="DONE SPEAKING")
            self.status_label.config(text="Press START RECORDING when ready")
            # Highlight voice button
            self.voice_btn.config(relief=tk.SUNKEN)
            self.text_btn.config(relief=tk.FLAT)

        elif mode == "text":
            self.mode_label.config(text="TEXT MODE — Type your thoughts, then ANALYSE",
                                   fg=TEXT_CLR)
            self.input_box.config(state=tk.NORMAL, bg="#1e2d40")
            self.input_box.focus_set()
            self.start_btn.config(state=tk.DISABLED, text="START RECORDING")
            self.done_btn.config(state=tk.NORMAL, text="ANALYSE TEXT", bg=TEXT_CLR)
            self.status_label.config(text="Type your feelings below and click ANALYSE TEXT")
            # Highlight text button
            self.text_btn.config(relief=tk.SUNKEN)
            self.voice_btn.config(relief=tk.FLAT)

    # ─── VOICE ACTIONS ───
    def start_recording(self):
        self.session = VoiceSession(self.model, self.ui_queue)
        self._set_input("")
        self._set_result("")
        self.start_btn.config(state=tk.DISABLED)
        self.done_btn.config(state=tk.NORMAL)
        threading.Thread(target=self.session.start_recording, daemon=True).start()

    def done_action(self):
        if self.current_mode == "voice":
            self.done_btn.config(state=tk.DISABLED)
            face_emotion = self.emo_var.get()
            threading.Thread(
                target=self.session.stop_and_analyse,
                args=(face_emotion,), daemon=True
            ).start()

        elif self.current_mode == "text":
            text = self.input_box.get("1.0", tk.END).strip()
            if not text:
                self.status_label.config(text="Please type something first.")
                return
            self.done_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Analysing your words...")
            threading.Thread(target=self._analyse_text, args=(text,), daemon=True).start()

    def _analyse_text(self, text):
        face_emotion = self.emo_var.get()
        result = analyze_sentiment(text)
        result["fused"] = fuse_signals(face_emotion, result["label"])
        log_result(result, "text")
        self.ui_queue.put(("result", result))

    # ─── HELPERS ───
    def clear_all(self):
        self._set_input("")
        self._set_result("")
        self.status_label.config(text="Ready.")
        if self.current_mode == "voice":
            self.start_btn.config(state=tk.NORMAL)
            self.done_btn.config(state=tk.DISABLED)
        elif self.current_mode == "text":
            self.done_btn.config(state=tk.NORMAL)

    def _set_input(self, text):
        self.input_box.config(state=tk.NORMAL)
        self.input_box.delete("1.0", tk.END)
        if text:
            self.input_box.insert(tk.END, text)
        if self.current_mode != "text":
            self.input_box.config(state=tk.DISABLED)

    def _set_result(self, text):
        self.result_box.config(state=tk.NORMAL)
        self.result_box.delete("1.0", tk.END)
        if text:
            self.result_box.insert(tk.END, text)
        self.result_box.config(state=tk.DISABLED)

    # ─── QUEUE PROCESSOR ───
    def _process_queue(self):
        try:
            while True:
                msg_type, data = self.ui_queue.get_nowait()

                if msg_type == "status":
                    self.status_label.config(text=data)

                elif msg_type == "transcript":
                    self._set_input(data)

                elif msg_type == "result":
                    if "error" in data:
                        self._set_result(data["error"])
                        self.status_label.config(text="Try again.")
                    else:
                        polarity_bar = self._polarity_bar(data["polarity"])
                        result_text = (
                            f"Sentiment   : {data['label']}  {polarity_bar}\n"
                            f"Polarity    : {data['polarity']:+.3f}   "
                            f"Subjectivity: {data['subjectivity']:.3f}\n"
                            f"Advice      : {data['advice']}\n\n"
                            f"EchoMirror  : {data['fused']}"
                        )
                        self._set_result(result_text)
                        self.status_label.config(
                            text="Analysis complete. Start a new session anytime.")

                    if self.current_mode == "voice":
                        self.start_btn.config(state=tk.NORMAL)
                    elif self.current_mode == "text":
                        self.done_btn.config(state=tk.NORMAL)

        except queue.Empty:
            pass

        self.root.after(100, self._process_queue)

    def _polarity_bar(self, polarity):
        """Visual polarity indicator"""
        filled = int((polarity + 1) / 2 * 10)
        bar = "█" * filled + "░" * (10 - filled)
        return f"[{bar}]"

    def run(self):
        self.root.mainloop()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run():
    print("[EchoMirror] Step 2 v3 — Voice + Text Sentiment")
    print(f"Device : {DEVICE.upper()}")
    print("Loading Whisper medium model...\n")

    model = whisper.load_model(WHISPER_MODEL, device=DEVICE)
    print(f"Whisper '{WHISPER_MODEL}' ready on {DEVICE.upper()}\n")

    app = EchoMirrorUI(model)
    app.run()


if __name__ == "__main__":
    run()