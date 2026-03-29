"""
EchoMirror - Step 1: Facial Emotion Detection Module (v3)
Improvements over v2:
  - Emotion smoothing (rolling average over last 5 detections — no jitter)
  - Confidence threshold (ignore weak detections below 40%)
  - Emotion stability counter (only update if emotion holds for 2+ frames)
  - Better face crop with padding for DeepFace accuracy
  - FPS counter displayed on screen
  - Emotion history log (last 5 emotions shown on screen)
  - DB integration — logs emotions to MemoryDB automatically
  - Exportable get_current_emotion() for Step 7 integration
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
from deepface import DeepFace
import threading
import time
from datetime import datetime
import numpy as np
from collections import deque, Counter

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CAMERA_INDEX         = 0
DETECTION_INTERVAL   = 0.8
DISPLAY_WIDTH        = 1280
DISPLAY_HEIGHT       = 720
LOW_LIGHT_THRESHOLD  = 80
CONFIDENCE_THRESHOLD = 40.0
SMOOTHING_WINDOW     = 5
STABILITY_REQUIRED   = 2

EMOTION_COLORS = {
    "happy":    (0, 255, 128),
    "sad":      (255, 100, 100),
    "angry":    (0, 0, 255),
    "fear":     (200, 0, 200),
    "surprise": (0, 200, 255),
    "disgust":  (0, 128, 0),
    "neutral":  (200, 200, 200),
}

EMOTION_AFFIRMATIONS = {
    "happy":    "Keep shining! You radiate positivity",
    "sad":      "It's okay to feel this way. You're not alone",
    "angry":    "Take a deep breath. You've got this",
    "fear":     "You are braver than you believe",
    "surprise": "Life is full of wonders!",
    "disgust":  "Your feelings are valid. Be kind to yourself",
    "neutral":  "A calm mind is a powerful mind",
}

EMOTION_EMOJIS = {
    "happy": "HAPPY", "sad": "SAD", "angry": "ANGRY",
    "fear": "FEAR", "surprise": "SURPRISE", "disgust": "DISGUST", "neutral": "NEUTRAL"
}


# ─────────────────────────────────────────
# LOW LIGHT ENHANCEMENT
# ─────────────────────────────────────────
def enhance_low_light(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    gamma = 1.8
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    l_enhanced = cv2.LUT(l_enhanced, table)
    enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def get_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def get_nearest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


# ─────────────────────────────────────────
# EMOTION DETECTOR v3
# ─────────────────────────────────────────
class EmotionDetector:
    def __init__(self, use_db=True):
        self.current_emotion    = "neutral"
        self.current_scores     = {}
        self.current_confidence = 0.0
        self.affirmation        = EMOTION_AFFIRMATIONS["neutral"]
        self.last_detection_time = 0
        self.lock               = threading.Lock()
        self.is_low_light       = False
        self.emotion_history    = deque(maxlen=SMOOTHING_WINDOW)
        self.recent_emotions    = deque(maxlen=5)
        self.fps                = 0
        self._frame_times       = deque(maxlen=30)

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.db = None
        if use_db:
            try:
                from database import MemoryDB
                self.db = MemoryDB()
                print("[EmotionDetector] MemoryDB connected")
            except Exception as e:
                print(f"[EmotionDetector] DB not available: {e}")

    def analyze_emotion(self, frame):
        try:
            result = DeepFace.analyze(frame, actions=["emotion"],
                                      enforce_detection=False, silent=True)
            if not result:
                return

            dominant   = result[0]["dominant_emotion"]
            scores     = result[0]["emotion"]
            confidence = scores.get(dominant, 0.0)

            if confidence < CONFIDENCE_THRESHOLD:
                print(f"[DeepFace] Low confidence ({confidence:.1f}%) skipped")
                return

            self.emotion_history.append(dominant)
            counts = Counter(self.emotion_history)
            stable_emotion = counts.most_common(1)[0][0]
            stable_count   = counts.most_common(1)[0][1]

            if stable_count >= STABILITY_REQUIRED:
                with self.lock:
                    if stable_emotion != self.current_emotion:
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                        top_str = " | ".join([f"{e}: {s:.1f}%" for e, s in top])
                        print(f"[{ts}] {stable_emotion.upper():10s} ({confidence:.1f}%) -> {top_str}")
                        if self.db:
                            try:
                                self.db.log_emotion(stable_emotion, confidence / 100.0)
                            except Exception:
                                pass
                        self.recent_emotions.append({
                            "emotion": stable_emotion,
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "conf": confidence
                        })
                    self.current_emotion    = stable_emotion
                    self.current_scores     = scores
                    self.current_confidence = confidence
                    self.affirmation        = EMOTION_AFFIRMATIONS.get(stable_emotion, "")

        except Exception as e:
            print(f"[DeepFace Error] {e}")

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_neighbors = 3 if self.is_low_light else 5
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(60, 60)
        )
        return faces

    def update_fps(self):
        now = time.time()
        self._frame_times.append(now)
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            self.fps = len(self._frame_times) / elapsed if elapsed > 0 else 0

    def get_state(self):
        with self.lock:
            return (self.current_emotion, self.current_scores,
                    self.affirmation, self.current_confidence)

    def get_current_emotion(self):
        """Clean interface for Step 7 integration"""
        with self.lock:
            return self.current_emotion, self.current_confidence


# ─────────────────────────────────────────
# OVERLAY v3
# ─────────────────────────────────────────
def draw_overlay(frame, nearest_face, emotion, scores, affirmation,
                 confidence, is_low_light, brightness, fps, recent_emotions):
    h, w = frame.shape[:2]
    color = EMOTION_COLORS.get(emotion, (200, 200, 200))

    # Face bounding box with glow effect
    if nearest_face is not None:
        x, y, fw, fh = nearest_face
        for thickness, c_mult in [(5, 0.2), (3, 0.5), (2, 1.0)]:
            c = tuple(int(ch * c_mult) for ch in color)
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), c, thickness)
        label = f"{emotion.upper()}  {confidence:.0f}%"
        cv2.putText(frame, label, (x, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        proximity = (fw * fh) / (w * h)
        dist = "Very Close" if proximity > 0.15 else "Close" if proximity > 0.07 else "Far"
        cv2.putText(frame, f"[{dist}]", (x, y + fh + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    # Top-left emotion box with confidence bar
    cv2.rectangle(frame, (10, 10), (370, 80), (20, 20, 30), -1)
    cv2.rectangle(frame, (10, 10), (370, 80), color, 2)
    cv2.putText(frame, f"Emotion: {emotion.upper()}", (20, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
    bar_len = int((confidence / 100) * 330)
    cv2.rectangle(frame, (20, 58), (350, 72), (40, 40, 40), -1)
    cv2.rectangle(frame, (20, 58), (20 + bar_len, 72), color, -1)
    cv2.putText(frame, f"{confidence:.0f}%", (355, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # Low light badge
    if is_low_light:
        cv2.rectangle(frame, (10, 86), (310, 110), (20, 20, 50), -1)
        cv2.putText(frame, f"Low Light Mode | Brightness: {int(brightness)}",
                    (15, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 180, 255), 1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)

    # Right side score bars
    if scores:
        bar_x = w - 230
        cv2.rectangle(frame, (bar_x - 170, 5), (w - 5, 225), (12, 12, 20), -1)
        cv2.putText(frame, "Emotion Scores", (bar_x - 160, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        for i, (emo, score) in enumerate(
                sorted(scores.items(), key=lambda x: x[1], reverse=True)):
            bar_len = int((score / 100) * 185)
            bar_color = EMOTION_COLORS.get(emo, (150, 150, 150))
            yp = 35 + i * 27
            cv2.rectangle(frame, (bar_x, yp), (bar_x + 185, yp + 16), (40, 40, 40), -1)
            cv2.rectangle(frame, (bar_x, yp), (bar_x + bar_len, yp + 16), bar_color, -1)
            is_dom = (emo == emotion)
            cv2.putText(frame, f"{emo[:7]:7s} {score:5.1f}%",
                        (bar_x - 165, yp + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255) if is_dom else (170, 170, 170),
                        2 if is_dom else 1)

    # Bottom history bar
    cv2.rectangle(frame, (0, h - 80), (w, h), (10, 10, 15), -1)
    if recent_emotions:
        cv2.putText(frame, "History:", (15, h - 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1)
        for i, entry in enumerate(list(recent_emotions)):
            ec = EMOTION_COLORS.get(entry["emotion"], (150, 150, 150))
            txt = f"{entry['time']} {entry['emotion'][:5].upper()} {entry['conf']:.0f}%"
            cv2.putText(frame, txt, (90 + i * 195, h - 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, ec, 1)

    # Affirmation text
    cv2.putText(frame, affirmation, (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1)

    return frame


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    detector = EmotionDetector(use_db=True)
    print("[EchoMirror] Emotion Detection v3")
    print("  Smoothing window     : 5 frames")
    print("  Confidence threshold : 40%")
    print("  Stability filter     : 2 frames")
    print("  Low light mode       : AUTO")
    print("  DB logging           : ON")
    print("  Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        detector.update_fps()

        brightness = get_brightness(frame)
        detector.is_low_light = brightness < LOW_LIGHT_THRESHOLD

        proc_frame = enhance_low_light(frame) if detector.is_low_light else frame.copy()

        all_faces    = detector.detect_faces(proc_frame)
        nearest_face = get_nearest_face(all_faces)

        if nearest_face is not None:
            now = time.time()
            if now - detector.last_detection_time >= DETECTION_INTERVAL:
                detector.last_detection_time = now
                x, y, fw, fh = nearest_face
                pad = 30
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(proc_frame.shape[1], x + fw + pad)
                y2 = min(proc_frame.shape[0], y + fh + pad)
                face_crop = proc_frame[y1:y2, x1:x2]
                threading.Thread(
                    target=detector.analyze_emotion,
                    args=(face_crop,), daemon=True
                ).start()
        else:
            cv2.putText(frame,
                "No face detected — move closer or improve lighting",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 255), 2)

        emotion, scores, affirmation, confidence = detector.get_state()
        frame = draw_overlay(
            frame, nearest_face, emotion, scores, affirmation,
            confidence, detector.is_low_light, brightness,
            detector.fps, detector.recent_emotions
        )

        cv2.imshow("EchoMirror - Emotion Detection v3", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[EchoMirror] Stopped.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()