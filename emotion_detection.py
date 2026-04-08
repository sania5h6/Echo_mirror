"""
EchoMirror - Step 1: Facial Emotion Detection Module (v4.1)
Improvements over v3:
  - DNN-based face detector (Caffe model) — far more accurate than Haar cascades
  - Falls back to Haar if DNN model not found
  - Weighted emotion smoothing with exponential decay (recent frames matter more)
  - Adaptive detection interval — faster when emotion is changing, slower when stable
  - Proportional face ROI padding (15% of face size, not fixed 30px)
  - Brightness & low-light flags now logged to DB correctly
  - Thread-safe face crop (deep copy before handing to thread)
  - Face tracking delta — skips re-detection when face barely moved
  - Micro-expression detection — detects brief emotion spikes
  - Transition animation on emotion change (smooth color fade)
  - Pulse glow effect on face bounding box
  - Graceful camera release with try/finally
  - Richer HUD: separate panels, anti-aliased text overlay, gradient bars
  - get_current_emotion() unchanged for Step 7 compatibility

v4.1 — Conversation-Level Emotion Resolution:
  - Two-tier emotion system: DISPLAY (fast/responsive) vs CONVERSATION (stable)
  - EmotionResolver: confidence-weighted voting over a 10s sliding window
  - Emotion group hysteresis: switching from 'sad' to 'angry' (same negative group)
    requires less evidence than switching from 'sad' to 'happy'
  - Emotion lock: after a switch, holds for 3 seconds minimum (anti-jitter)
  - Minimum stability threshold (0.50) + 2s cooldown between display switches
  - get_conversation_emotion() API for Step 3/7 integration
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
import math

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CAMERA_INDEX               = 0
BASE_DETECTION_INTERVAL    = 0.5     # Base interval (adaptive: 0.3–0.8s)
FAST_DETECTION_INTERVAL    = 0.3     # When emotion is unstable
SLOW_DETECTION_INTERVAL    = 0.8     # When emotion is stable
DISPLAY_WIDTH              = 1280
DISPLAY_HEIGHT             = 720
LOW_LIGHT_THRESHOLD        = 80
CONFIDENCE_THRESHOLD       = 40.0    # Slightly lower — DNN is more reliable
SMOOTHING_WINDOW           = 11      # Larger window for smoother display output
STABILITY_REQUIRED         = 4       # Frames emotion must hold before updating display
STABILITY_MIN_SCORE        = 0.50    # Weighted stability must exceed this to switch
DISPLAY_SWITCH_COOLDOWN    = 2.0     # Seconds — minimum gap between display emotion changes
FACE_MOVE_THRESHOLD        = 25      # Pixels — skip re-detect if face barely moved
MICRO_EXPRESSION_WINDOW    = 3       # Frames to detect brief spikes
MICRO_EXPRESSION_MIN_CONF  = 70.0    # Minimum confidence for micro-expression
DNN_CONFIDENCE_THRESHOLD   = 0.55    # DNN face detector confidence cutoff

# ── Conversation-Level Emotion Resolution ──
CONVERSATION_WINDOW        = 10.0    # Seconds — sliding window for conversation emotion
EMOTION_LOCK_SECONDS       = 3.0     # After switching, hold for at least this long
EMOTION_SWITCH_THRESHOLD   = 0.55    # Confidence-weighted vote share to switch
CROSS_GROUP_THRESHOLD      = 0.65    # Higher threshold to cross emotion groups

# Emotion groups — switching within a group is easier than across groups
NEGATIVE_EMOTIONS = {"sad", "angry", "fear", "disgust"}
POSITIVE_EMOTIONS = {"happy", "surprise"}
NEUTRAL_EMOTIONS  = {"neutral"}

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

# Decay weights for smoothing window — most recent frame = highest weight
DECAY_WEIGHTS = [math.exp(-0.3 * i) for i in range(SMOOTHING_WINDOW)]
DECAY_WEIGHTS.reverse()  # Index 0 = oldest, last = newest


def _get_emotion_group(emotion):
    """Return which group an emotion belongs to."""
    if emotion in NEGATIVE_EMOTIONS:
        return "negative"
    elif emotion in POSITIVE_EMOTIONS:
        return "positive"
    return "neutral"


# ─────────────────────────────────────────
# LOW LIGHT ENHANCEMENT
# ─────────────────────────────────────────
def enhance_low_light(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    gamma = 1.8
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in range(256)]).astype("uint8")
    l_enhanced = cv2.LUT(l_enhanced, table)
    enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def get_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def get_nearest_face(faces):
    """Select the largest (nearest) face from a list of (x, y, w, h) tuples."""
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def lerp_color(c1, c2, t):
    """Linearly interpolate between two BGR colors."""
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


# ─────────────────────────────────────────
# DNN FACE DETECTOR (Caffe model)
# ─────────────────────────────────────────
class FaceDetectorDNN:
    """
    Uses OpenCV's pre-trained Caffe DNN face detector.
    Falls back to Haar cascade if model files are missing.
    """
    def __init__(self):
        self.use_dnn = False
        self.net = None
        self.haar = None

        # Try loading DNN model
        proto_path = cv2.data.haarcascades + "../res10_300x300_ssd_iter_140000.caffemodel"
        config_path = cv2.data.haarcascades + "../deploy.prototxt"

        # Check common locations
        dnn_paths = [
            (config_path, proto_path),
            ("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel"),
        ]

        for prototxt, model in dnn_paths:
            if os.path.exists(prototxt) and os.path.exists(model):
                try:
                    self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
                    self.use_dnn = True
                    print("[FaceDetector] DNN Caffe model loaded ✓")
                    break
                except Exception as e:
                    print(f"[FaceDetector] DNN load failed: {e}")

        if not self.use_dnn:
            self.haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            print("[FaceDetector] Using Haar cascade (fallback)")

    def detect(self, frame, is_low_light=False):
        """Returns list of (x, y, w, h) tuples for detected faces."""
        if self.use_dnn:
            return self._detect_dnn(frame)
        else:
            return self._detect_haar(frame, is_low_light)

    def _detect_dnn(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > DNN_CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                # Clamp to frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                fw = x2 - x1
                fh = y2 - y1
                if fw > 40 and fh > 40:  # Min face size
                    faces.append((x1, y1, fw, fh))
        return faces

    def _detect_haar(self, frame, is_low_light=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_neighbors = 3 if is_low_light else 5
        faces = self.haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(60, 60)
        )
        return list(faces) if len(faces) > 0 else []


# ─────────────────────────────────────────
# EMOTION DETECTOR v4
# ─────────────────────────────────────────
class EmotionDetector:
    def __init__(self, use_db=True):
        # ── Display-level state (per-frame, for HUD) ──
        self.current_emotion       = "neutral"
        self.previous_emotion      = "neutral"
        self.current_scores        = {}
        self.current_confidence    = 0.0
        self.affirmation           = EMOTION_AFFIRMATIONS["neutral"]
        self.last_detection_time   = 0
        self.lock                  = threading.Lock()
        self.is_low_light          = False
        self.brightness            = 255.0
        self.emotion_history       = deque(maxlen=SMOOTHING_WINDOW)
        self.raw_emotion_history   = deque(maxlen=MICRO_EXPRESSION_WINDOW)
        self.recent_emotions       = deque(maxlen=5)
        self.fps                   = 0
        self._frame_times          = deque(maxlen=30)
        self.last_face_pos         = None       # For face tracking delta
        self.emotion_change_time   = 0          # For transition animation
        self.micro_expression      = None       # Detected micro-expression
        self._detection_interval   = BASE_DETECTION_INTERVAL
        self._stable_count         = 0          # How long current emotion held

        # ── Conversation-level state (stable, for AI) ──
        self._resolver_buffer      = deque(maxlen=200)  # Timestamped detections
        self._conv_emotion         = "neutral"
        self._conv_confidence      = 0.0
        self._conv_switch_time     = 0

        # Face detector (DNN or Haar)
        self.face_detector = FaceDetectorDNN()

        # DB integration
        self.db = None
        if use_db:
            try:
                from database import MemoryDB
                self.db = MemoryDB()
                print("[EmotionDetector] MemoryDB connected")
            except Exception as e:
                print(f"[EmotionDetector] DB not available: {e}")

    def _weighted_dominant_emotion(self):
        """
        Compute dominant emotion using exponential decay weights.
        More recent frames have higher influence.
        """
        if not self.emotion_history:
            return "neutral", 0

        history = list(self.emotion_history)
        weighted_counts = {}
        n = len(history)
        # Use last N weights matching history length
        weights = DECAY_WEIGHTS[-n:]

        for emo, weight in zip(history, weights):
            weighted_counts[emo] = weighted_counts.get(emo, 0) + weight

        dominant = max(weighted_counts, key=weighted_counts.get)
        # Compute stability: what fraction of total weight is the dominant emotion
        total_weight = sum(weighted_counts.values())
        stability = weighted_counts[dominant] / total_weight if total_weight > 0 else 0
        return dominant, stability

    def _detect_micro_expression(self, emotion, confidence):
        """
        Detect brief emotion spikes that differ from the stable emotion.
        A micro-expression is a high-confidence emotion that appears
        for 1-2 frames then vanishes — indicates suppressed emotion.
        """
        self.raw_emotion_history.append((emotion, confidence))
        if len(self.raw_emotion_history) < MICRO_EXPRESSION_WINDOW:
            return

        raw = list(self.raw_emotion_history)
        # Check if the middle frame has a different high-confidence emotion
        # while surrounding frames show the stable emotion
        if len(raw) >= 3:
            mid_emo, mid_conf = raw[-2]
            prev_emo, _ = raw[-3]
            curr_emo, _ = raw[-1]

            if (mid_emo != self.current_emotion and
                mid_conf >= MICRO_EXPRESSION_MIN_CONF and
                prev_emo == self.current_emotion and
                curr_emo == self.current_emotion):
                self.micro_expression = {
                    "emotion": mid_emo,
                    "confidence": mid_conf,
                    "time": datetime.now().strftime("%H:%M:%S"),
                }
                print(f"[MicroExpr] Detected: {mid_emo} ({mid_conf:.1f}%) "
                      f"while stable={self.current_emotion}")

    def _update_detection_interval(self):
        """
        Adaptive detection pacing:
        - Faster when emotion is changing (unstable)
        - Slower when stable to save CPU
        """
        if self._stable_count >= 8:
            self._detection_interval = SLOW_DETECTION_INTERVAL
        elif self._stable_count <= 2:
            self._detection_interval = FAST_DETECTION_INTERVAL
        else:
            self._detection_interval = BASE_DETECTION_INTERVAL

    def analyze_emotion(self, frame):
        try:
            result = DeepFace.analyze(frame, actions=["emotion"],
                                      enforce_detection=False, silent=True)
            if not result:
                return

            scores = result[0]["emotion"]

            # Use actual highest score, not DeepFace's label
            dominant   = max(scores, key=scores.get)
            confidence = scores[dominant]

            if confidence < CONFIDENCE_THRESHOLD:
                return

            # Micro-expression detection
            self._detect_micro_expression(dominant, confidence)

            # Add to smoothing window
            self.emotion_history.append(dominant)

            # Feed the conversation resolver (timestamped, confidence-weighted)
            self._resolver_feed(dominant, confidence)

            # Weighted smoothing for DISPLAY emotion
            stable_emotion, stability = self._weighted_dominant_emotion()
            # Also require raw count >= STABILITY_REQUIRED
            counts = Counter(self.emotion_history)
            raw_count = counts.get(stable_emotion, 0)

            # Strengthened gate: stability score + raw count + cooldown
            now = time.time()
            cooldown_ok = (now - self.emotion_change_time) >= DISPLAY_SWITCH_COOLDOWN

            if (raw_count >= STABILITY_REQUIRED and
                stability >= STABILITY_MIN_SCORE and
                cooldown_ok):
                with self.lock:
                    if stable_emotion != self.current_emotion:
                        self.previous_emotion = self.current_emotion
                        self.emotion_change_time = now
                        self._stable_count = 0

                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                        top_str = " | ".join([f"{e}: {s:.1f}%" for e, s in top])
                        print(f"[{ts}] {stable_emotion.upper():10s} "
                              f"({confidence:.1f}% | stab={stability:.2f}) -> {top_str}")

                        # Log to DB with brightness & low-light
                        if self.db:
                            try:
                                self.db.log_emotion(
                                    stable_emotion,
                                    confidence / 100.0,
                                    brightness=self.brightness,
                                    is_low_light=self.is_low_light
                                )
                            except Exception:
                                pass

                        self.recent_emotions.append({
                            "emotion": stable_emotion,
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "conf": confidence
                        })
                    else:
                        self._stable_count += 1

                    self.current_emotion    = stable_emotion
                    self.current_scores     = scores
                    self.current_confidence = confidence
                    self.affirmation        = EMOTION_AFFIRMATIONS.get(stable_emotion, "")

            # Update adaptive interval
            self._update_detection_interval()

        except Exception as e:
            print(f"[DeepFace Error] {e}")

    def detect_faces(self, frame):
        return self.face_detector.detect(frame, self.is_low_light)

    def has_face_moved(self, face):
        """Check if the face has moved enough to warrant re-detection."""
        if face is None or self.last_face_pos is None:
            return True
        x, y, w, h = face
        lx, ly, lw, lh = self.last_face_pos
        cx, cy = x + w // 2, y + h // 2
        lcx, lcy = lx + lw // 2, ly + lh // 2
        dist = math.sqrt((cx - lcx) ** 2 + (cy - lcy) ** 2)
        size_change = abs(w * h - lw * lh) / max(lw * lh, 1)
        return dist > FACE_MOVE_THRESHOLD or size_change > 0.15

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

    # ─── CONVERSATION-LEVEL RESOLVER ───

    def _resolver_feed(self, emotion, confidence):
        """Feed a detection to the conversation resolver."""
        with self.lock:
            self._resolver_buffer.append({
                "emotion": emotion,
                "confidence": confidence,
                "time": time.time(),
            })

    def get_current_emotion(self):
        """Clean interface for Step 7 integration (returns DISPLAY emotion)"""
        with self.lock:
            return self.current_emotion, self.current_confidence

    def get_conversation_emotion(self):
        """
        Returns the CONVERSATION-LEVEL resolved emotion.
        This is the emotion that the ConversationEngine / Step 7 should use.
        It's far more stable than the per-frame display emotion.

        Returns:
            (emotion: str, confidence: float, group: str)
            e.g. ("sad", 82.3, "negative")
        """
        with self.lock:
            resolved = self._resolve_conversation_emotion()
            return resolved

    def _resolve_conversation_emotion(self):
        """
        Two-tier emotion resolution for conversation:
        1. Collect all detections in the last CONVERSATION_WINDOW seconds
        2. Compute confidence-weighted vote for each emotion
        3. Apply emotion-group hysteresis (harder to cross groups)
        4. Apply emotion lock (hold after switch)
        """
        now = time.time()

        # Prune old entries outside the conversation window
        cutoff = now - CONVERSATION_WINDOW
        while self._resolver_buffer and self._resolver_buffer[0]["time"] < cutoff:
            self._resolver_buffer.popleft()

        if not self._resolver_buffer:
            return self._conv_emotion, self._conv_confidence, _get_emotion_group(self._conv_emotion)

        # Confidence-weighted vote
        weighted_votes = {}
        total_weight = 0
        for entry in self._resolver_buffer:
            emo = entry["emotion"]
            conf = entry["confidence"]
            weighted_votes[emo] = weighted_votes.get(emo, 0) + conf
            total_weight += conf

        if total_weight == 0:
            return self._conv_emotion, self._conv_confidence, _get_emotion_group(self._conv_emotion)

        # Normalize to vote shares
        vote_shares = {emo: w / total_weight for emo, w in weighted_votes.items()}
        top_emotion = max(vote_shares, key=vote_shares.get)
        top_share = vote_shares[top_emotion]

        # Avg confidence for the top emotion
        top_confs = [e["confidence"] for e in self._resolver_buffer if e["emotion"] == top_emotion]
        avg_confidence = sum(top_confs) / len(top_confs) if top_confs else 0

        # Check if we should switch
        current_group = _get_emotion_group(self._conv_emotion)
        new_group = _get_emotion_group(top_emotion)
        cross_group = (current_group != new_group)

        # Higher threshold for crossing groups (e.g. sad→happy)
        threshold = CROSS_GROUP_THRESHOLD if cross_group else EMOTION_SWITCH_THRESHOLD

        # Emotion lock — after a switch, hold for EMOTION_LOCK_SECONDS
        lock_expired = (now - self._conv_switch_time) >= EMOTION_LOCK_SECONDS

        if top_emotion != self._conv_emotion and top_share >= threshold and lock_expired:
            old = self._conv_emotion
            self._conv_emotion = top_emotion
            self._conv_confidence = avg_confidence
            self._conv_switch_time = now
            print(f"[ConvResolver] {old.upper()} → {top_emotion.upper()} "
                  f"(share={top_share:.2f}, conf={avg_confidence:.1f}%, "
                  f"{'CROSS-GROUP' if cross_group else 'same-group'})")
        elif top_emotion == self._conv_emotion:
            # Update confidence even if not switching
            self._conv_confidence = avg_confidence

        return self._conv_emotion, self._conv_confidence, _get_emotion_group(self._conv_emotion)


# ─────────────────────────────────────────
# OVERLAY v4
# ─────────────────────────────────────────
def draw_rounded_rect(frame, pt1, pt2, color, thickness, radius=12):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 4, (y2 - y1) // 4)

    if thickness == -1:
        # Filled
        cv2.rectangle(frame, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(frame, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(frame, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(frame, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(frame, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(frame, (x2 - r, y2 - r), r, color, -1)
    else:
        # Outline
        cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_gradient_bar(frame, x, y, width, height, value, max_val, color):
    """Draw a horizontal bar with gradient fill."""
    fill_w = int((value / max_val) * width) if max_val > 0 else 0
    fill_w = max(0, min(fill_w, width))

    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (30, 30, 40), -1)
    # Filled portion with slight gradient
    if fill_w > 0:
        for i in range(fill_w):
            t = i / max(fill_w, 1)
            c = tuple(int(ch * (0.6 + 0.4 * t)) for ch in color)
            cv2.line(frame, (x + i, y), (x + i, y + height), c, 1)


def draw_overlay(frame, nearest_face, emotion, scores, affirmation,
                 confidence, is_low_light, brightness, fps, recent_emotions,
                 detector=None):
    h, w = frame.shape[:2]
    target_color = EMOTION_COLORS.get(emotion, (200, 200, 200))

    # Smooth color transition on emotion change
    if detector and detector.emotion_change_time > 0:
        elapsed = time.time() - detector.emotion_change_time
        transition_t = min(elapsed / 0.6, 1.0)  # 600ms transition
        prev_color = EMOTION_COLORS.get(detector.previous_emotion, (200, 200, 200))
        color = lerp_color(prev_color, target_color, transition_t)
    else:
        color = target_color

    # Face box with animated pulse glow
    if nearest_face is not None:
        x, y, fw, fh = nearest_face
        pulse = 0.5 + 0.5 * math.sin(time.time() * 3.0)  # Pulsing 0–1

        # Outer glow (pulsing)
        glow_color = tuple(int(ch * 0.15 * pulse) for ch in color)
        cv2.rectangle(frame, (x - 4, y - 4), (x + fw + 4, y + fh + 4), glow_color, 6)
        # Mid glow
        mid_color = tuple(int(ch * (0.4 + 0.2 * pulse)) for ch in color)
        cv2.rectangle(frame, (x - 2, y - 2), (x + fw + 2, y + fh + 2), mid_color, 3)
        # Inner border
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

        # Emotion label above face
        label = f"{emotion.upper()}  {confidence:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_x = x + (fw - tw) // 2  # Center above face
        label_y = y - 14
        # Label background
        cv2.rectangle(frame, (label_x - 6, label_y - th - 4),
                      (label_x + tw + 6, label_y + 4), (10, 10, 20), -1)
        cv2.putText(frame, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Distance indicator
        proximity = (fw * fh) / (w * h)
        dist = "Very Close" if proximity > 0.15 else "Close" if proximity > 0.07 else "Far"
        cv2.putText(frame, f"[{dist}]", (x, y + fh + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

    # ── Top-left panel: emotion + confidence ──
    panel_w = 380
    panel_h = 105  # Taller to fit conversation emotion
    # Semi-transparent panel background
    overlay = frame.copy()
    draw_rounded_rect(overlay, (10, 10), (panel_w, panel_h), (15, 15, 25), -1, 10)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    draw_rounded_rect(frame, (10, 10), (panel_w, panel_h), color, 2, 10)

    cv2.putText(frame, f"Display: {emotion.upper()}", (22, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
    # Gradient confidence bar
    draw_gradient_bar(frame, 22, 46, 330, 12, confidence, 100, color)
    cv2.putText(frame, f"{confidence:.0f}%", (358, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Conversation-resolved emotion (what the AI actually uses)
    if detector:
        conv_emo, conv_conf, conv_group = detector.get_conversation_emotion()
        conv_color = EMOTION_COLORS.get(conv_emo, (200, 200, 200))
        group_tag = f"[{conv_group}]"
        cv2.putText(frame, f"Conv: {conv_emo.upper()} {group_tag}", (22, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, conv_color, 2)
        draw_gradient_bar(frame, 22, 84, 330, 10, conv_conf, 100, conv_color)
        cv2.putText(frame, f"{conv_conf:.0f}%", (358, 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, conv_color, 1)

    # ── Low light badge ──
    badge_y = panel_h + 8
    if is_low_light:
        cv2.rectangle(frame, (10, badge_y), (320, badge_y + 24), (15, 15, 40), -1)
        cv2.putText(frame, f"Low Light | Brightness: {int(brightness)}",
                    (16, badge_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 160, 255), 1)

    # ── Micro-expression indicator ──
    if detector and detector.micro_expression:
        me = detector.micro_expression
        me_age = time.time() - detector.emotion_change_time
        if me_age < 5.0:  # Show for 5 seconds
            me_color = EMOTION_COLORS.get(me["emotion"], (200, 200, 200))
            me_y = badge_y + (28 if is_low_light else 0)
            cv2.rectangle(frame, (10, me_y), (360, me_y + 24), (20, 10, 30), -1)
            cv2.putText(frame, f"Micro: {me['emotion'].upper()} ({me['confidence']:.0f}%) at {me['time']}",
                        (16, me_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, me_color, 1)

    # ── FPS + detection mode ──
    mode_str = "DNN" if detector and detector.face_detector.use_dnn else "Haar"
    cv2.putText(frame, f"FPS: {fps:.0f} | {mode_str}", (w - 160, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1)

    # Adaptive interval indicator
    if detector:
        interval = detector._detection_interval
        interval_color = (0, 200, 100) if interval <= 0.4 else (200, 200, 0) if interval <= 0.6 else (100, 100, 100)
        cv2.putText(frame, f"Scan: {interval:.1f}s", (w - 160, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, interval_color, 1)

    # ── Right side: emotion score bars (sorted) ──
    if scores:
        bar_x = w - 230
        panel_top = 58
        panel_bot = panel_top + 215
        # Panel background
        overlay = frame.copy()
        draw_rounded_rect(overlay, (bar_x - 172, panel_top - 5),
                          (w - 5, panel_bot), (10, 10, 18), -1, 8)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        cv2.putText(frame, "Emotion Scores", (bar_x - 160, panel_top + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        for i, (emo, score) in enumerate(
                sorted(scores.items(), key=lambda x: x[1], reverse=True)):
            bc = EMOTION_COLORS.get(emo, (150, 150, 150))
            yp = panel_top + 24 + i * 27
            # Gradient bar for each emotion
            draw_gradient_bar(frame, bar_x, yp, 185, 16, score, 100, bc)
            is_dom = (emo == emotion)
            label_color = (255, 255, 255) if is_dom else (160, 160, 160)
            thickness = 2 if is_dom else 1
            cv2.putText(frame, f"{emo[:7]:7s} {score:5.1f}%",
                        (bar_x - 165, yp + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, thickness)

    # ── Bottom: history bar ──
    cv2.rectangle(frame, (0, h - 80), (w, h), (8, 8, 12), -1)
    if recent_emotions:
        cv2.putText(frame, "History:", (15, h - 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1)
        for i, entry in enumerate(list(recent_emotions)):
            ec = EMOTION_COLORS.get(entry["emotion"], (150, 150, 150))
            txt = f"{entry['time']} {entry['emotion'][:5].upper()} {entry['conf']:.0f}%"
            cv2.putText(frame, txt, (90 + i * 200, h - 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, ec, 1)

    # ── Affirmation ──
    cv2.putText(frame, affirmation, (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1)

    return frame


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def _open_camera(index, width, height):
    """Open camera with DirectShow (stable on Windows), fallback to default."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if cap.isOpened():
        return cap, "DirectShow"
    cap.release()
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if cap.isOpened():
        return cap, "Default"
    return None, None


def run():
    cap, backend = _open_camera(CAMERA_INDEX, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    if cap is None:
        print("[ERROR] Cannot open camera. Check if another app is using it.")
        return

    detector = EmotionDetector(use_db=True)
    print("[EchoMirror] Emotion Detection v4.1")
    print(f"  Camera backend       : {backend}")
    print(f"  Face detector        : {'DNN (Caffe)' if detector.face_detector.use_dnn else 'Haar cascade'}")
    print(f"  Smoothing window     : {SMOOTHING_WINDOW} frames (weighted)")
    print(f"  Confidence threshold : {CONFIDENCE_THRESHOLD}%")
    print(f"  Stability filter     : {STABILITY_REQUIRED} frames (min {STABILITY_MIN_SCORE:.0%})")
    print(f"  Display cooldown     : {DISPLAY_SWITCH_COOLDOWN}s")
    print(f"  Adaptive interval    : {FAST_DETECTION_INTERVAL}s – {SLOW_DETECTION_INTERVAL}s")
    print(f"  Face move threshold  : {FACE_MOVE_THRESHOLD}px")
    print("  Low light mode       : AUTO")
    print("  DB logging           : ON (with brightness)")
    print("  Micro-expressions    : ON")
    print(f"  \u2500\u2500 Conversation Resolver \u2500\u2500")
    print(f"  Conv window          : {CONVERSATION_WINDOW}s")
    print(f"  Emotion lock         : {EMOTION_LOCK_SECONDS}s")
    print(f"  Switch threshold     : {EMOTION_SWITCH_THRESHOLD:.0%} (same-group) / {CROSS_GROUP_THRESHOLD:.0%} (cross-group)")
    print("  Press 'q' to quit\n")

    frame_fail_count = 0
    MAX_FRAME_FAILS = 30
    camera_recoveries = 0
    MAX_RECOVERIES = 3

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                frame_fail_count += 1
                if frame_fail_count >= MAX_FRAME_FAILS:
                    if camera_recoveries < MAX_RECOVERIES:
                        camera_recoveries += 1
                        print(f"[Camera] Lost connection — recovering "
                              f"(attempt {camera_recoveries}/{MAX_RECOVERIES})...")
                        cap.release()
                        time.sleep(1)
                        cap, backend = _open_camera(CAMERA_INDEX, DISPLAY_WIDTH, DISPLAY_HEIGHT)
                        frame_fail_count = 0
                        if cap is not None:
                            print(f"[Camera] Recovered via {backend} ✓")
                            continue
                    print(f"[ERROR] Camera unrecoverable after {MAX_RECOVERIES} attempts.")
                    break
                if frame_fail_count % 10 == 1:
                    print(f"[WARN] Frame read failed ({frame_fail_count}/{MAX_FRAME_FAILS})...")
                time.sleep(0.1)
                continue
            frame_fail_count = 0

            frame = cv2.flip(frame, 1)
            detector.update_fps()

            brightness = get_brightness(frame)
            detector.brightness = brightness
            detector.is_low_light = brightness < LOW_LIGHT_THRESHOLD
            proc_frame = enhance_low_light(frame) if detector.is_low_light else frame.copy()

            all_faces    = detector.detect_faces(proc_frame)
            nearest_face = get_nearest_face(all_faces)

            if nearest_face is not None:
                now = time.time()
                should_detect = (now - detector.last_detection_time >= detector._detection_interval)
                face_moved = detector.has_face_moved(nearest_face)

                if should_detect or face_moved:
                    detector.last_detection_time = now
                    detector.last_face_pos = nearest_face

                    x, y, fw, fh = nearest_face
                    # Proportional padding (15% of face size)
                    pad_x = int(fw * 0.15)
                    pad_y = int(fh * 0.15)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(proc_frame.shape[1], x + fw + pad_x)
                    y2 = min(proc_frame.shape[0], y + fh + pad_y)
                    # Deep copy for thread safety
                    face_crop = proc_frame[y1:y2, x1:x2].copy()
                    threading.Thread(
                        target=detector.analyze_emotion,
                        args=(face_crop,), daemon=True
                    ).start()
            else:
                detector.last_face_pos = None
                cv2.putText(frame,
                    "No face detected — move closer or improve lighting",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (60, 60, 200), 2)

            emotion, scores, affirmation, confidence = detector.get_state()
            frame = draw_overlay(
                frame, nearest_face, emotion, scores, affirmation,
                confidence, detector.is_low_light, brightness,
                detector.fps, detector.recent_emotions,
                detector=detector
            )

            cv2.imshow("EchoMirror - Emotion Detection v4", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[EchoMirror] Stopped.")
                break

    except KeyboardInterrupt:
        print("\n[EchoMirror] Interrupted — shutting down.")
    except Exception as e:
        print(f"\n[EchoMirror] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[EchoMirror] Camera released. Goodbye.")


if __name__ == "__main__":
    run()