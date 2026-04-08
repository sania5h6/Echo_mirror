"""
EchoMirror - Explainable AI Module (XAI)
Provides transparency into:
  1. WHY a particular emotion was detected (score breakdown)
  2. WHAT facial cues contributed (expression analysis)
  3. HOW face + voice + text signals were fused
  4. WHY the AI responded the way it did
  5. Confidence explanations for the user

Referenced in: Slide 9, 10, 15 of the project proposal.
"""

import os
from datetime import datetime
from collections import deque


# ─────────────────────────────────────────
# EMOTION EXPLANATION MAPPINGS
# ─────────────────────────────────────────
FACIAL_CUE_MAP = {
    "happy":    ["raised cheeks", "smile lines around eyes", "upturned mouth corners"],
    "sad":      ["lowered brow", "downturned mouth", "drooping eyelids"],
    "angry":    ["furrowed brows", "tightened jaw", "narrowed eyes"],
    "fear":     ["widened eyes", "raised eyebrows", "slightly open mouth"],
    "surprise": ["raised eyebrows", "wide open eyes", "open mouth"],
    "disgust":  ["wrinkled nose", "raised upper lip", "narrowed eyes"],
    "neutral":  ["relaxed facial muscles", "even brow position", "calm expression"],
}

EMOTION_DESCRIPTIONS = {
    "happy":    "Your facial expression shows signs of happiness",
    "sad":      "Your facial expression indicates sadness",
    "angry":    "Your facial expression suggests anger or frustration",
    "fear":     "Your facial expression shows signs of anxiety or fear",
    "surprise": "Your facial expression indicates surprise",
    "disgust":  "Your facial expression suggests discomfort",
    "neutral":  "Your facial expression appears calm and neutral",
}

SIGNAL_WEIGHT_LABELS = {
    "face_dominant":  "Your facial expression was the primary signal",
    "voice_dominant": "Your voice tone was the primary signal",
    "text_dominant":  "Your word choice was the primary signal",
    "face_voice":     "Both your expression and voice tone agree",
    "face_text":      "Both your expression and word choice agree",
    "all_agree":      "Your face, voice, and text all point the same direction",
    "conflict":       "Your face and words seem to tell different stories",
}


# ─────────────────────────────────────────
# EMOTION EXPLAINER
# ─────────────────────────────────────────
class EmotionExplainer:
    """
    Generates human-readable explanations for detected emotions.
    Used by the HUD overlay and injected into the AI system prompt.
    """

    def __init__(self):
        self.explanation_history = deque(maxlen=10)
        self._last_explanation = None

    def explain_detection(self, emotion: str, scores: dict,
                          confidence: float) -> dict:
        """
        Generate a full explanation for why an emotion was detected.

        Args:
            emotion: The detected dominant emotion
            scores: Dict of all emotion scores from DeepFace
            confidence: The confidence of the dominant emotion

        Returns:
            dict with keys: summary, cues, breakdown, confidence_level, raw
        """
        if not scores:
            return self._empty_explanation(emotion)

        # Sort scores for ranking
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_emotion = ranked[0][0]
        top_score = ranked[0][1]

        # Confidence level interpretation
        if confidence >= 85:
            conf_level = "very high"
            conf_note = "The system is highly confident in this reading"
        elif confidence >= 65:
            conf_level = "high"
            conf_note = "The system is fairly confident in this reading"
        elif confidence >= 45:
            conf_level = "moderate"
            conf_note = "The reading has moderate certainty — subtle expression"
        else:
            conf_level = "low"
            conf_note = "The reading has low certainty — expression is ambiguous"

        # Facial cues explanation
        cues = FACIAL_CUE_MAP.get(emotion, ["subtle facial tension"])
        cue_text = f"Detected cues: {', '.join(cues[:2])}"

        # Score breakdown (top 3)
        breakdown_items = []
        for emo, score in ranked[:3]:
            pct = f"{score:.1f}%"
            if emo == emotion:
                breakdown_items.append(f"**{emo}: {pct}** (dominant)")
            else:
                breakdown_items.append(f"{emo}: {pct}")
        breakdown_text = " | ".join(breakdown_items)

        # Secondary emotion influence
        secondary_note = ""
        if len(ranked) >= 2 and ranked[1][1] > 20:
            sec_emo = ranked[1][0]
            sec_score = ranked[1][1]
            if sec_score > top_score * 0.4:
                secondary_note = (f"Note: {sec_emo} ({sec_score:.0f}%) "
                                  f"is also significantly present — mixed expression")

        # Summary sentence
        desc = EMOTION_DESCRIPTIONS.get(emotion, f"Detected: {emotion}")
        summary = f"{desc} ({confidence:.0f}% confidence). {cue_text}."

        explanation = {
            "summary": summary,
            "cues": cues,
            "cue_text": cue_text,
            "breakdown": breakdown_text,
            "confidence_level": conf_level,
            "confidence_note": conf_note,
            "secondary_note": secondary_note,
            "emotion": emotion,
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "raw_scores": scores,
        }

        self._last_explanation = explanation
        self.explanation_history.append(explanation)
        return explanation

    def explain_fusion(self, face_emotion: str, face_confidence: float,
                       text_sentiment: str, text_polarity: float,
                       voice_sentiment: str = None) -> dict:
        """
        Explain how multimodal signals were combined.

        Returns:
            dict with keys: fusion_summary, agreement, dominant_signal, conflict
        """
        # Determine agreement
        face_is_negative = face_emotion in ("sad", "angry", "fear", "disgust")
        text_is_negative = text_polarity < -0.1
        face_is_positive = face_emotion in ("happy", "surprise")
        text_is_positive = text_polarity > 0.1

        if face_is_negative and text_is_negative:
            agreement = "strong_negative"
            agree_text = "Both your expression and words indicate distress"
            dominant = "all_agree"
        elif face_is_positive and text_is_positive:
            agreement = "strong_positive"
            agree_text = "Both your expression and words radiate positivity"
            dominant = "all_agree"
        elif face_is_negative and text_is_positive:
            agreement = "conflict_face_neg"
            agree_text = ("Your face shows distress but your words are positive "
                          "— you might be putting on a brave face")
            dominant = "conflict"
        elif face_is_positive and text_is_negative:
            agreement = "conflict_face_pos"
            agree_text = ("You're smiling but your words carry weight "
                          "— there may be more beneath the surface")
            dominant = "conflict"
        else:
            agreement = "neutral"
            agree_text = "Your signals are balanced and calm"
            dominant = "face_dominant" if face_confidence > 60 else "text_dominant"

        # Build fusion summary
        parts = [f"Face: {face_emotion} ({face_confidence:.0f}%)"]
        parts.append(f"Text: {text_sentiment} ({text_polarity:+.2f})")
        if voice_sentiment:
            parts.append(f"Voice: {voice_sentiment}")
        signal_line = " | ".join(parts)

        fusion = {
            "fusion_summary": f"{agree_text}. [{signal_line}]",
            "agreement": agreement,
            "agreement_text": agree_text,
            "dominant_signal": dominant,
            "dominant_label": SIGNAL_WEIGHT_LABELS.get(dominant, ""),
            "is_conflict": "conflict" in agreement,
            "signal_line": signal_line,
        }
        return fusion

    def explain_response(self, emotion: str, sentiment: str,
                         polarity: float, memory_context: str,
                         has_wisdom: bool, has_exercise: bool) -> str:
        """
        Generate an explanation of WHY EchoMirror responded the way it did.
        This is shown to the user for transparency.
        """
        reasons = []

        # Emotion influence
        reasons.append(f"Your detected emotion ({emotion}) shaped the tone")

        # Sentiment influence
        if polarity < -0.3:
            reasons.append("your words carry deep negative weight, "
                           "so I'm being extra gentle")
        elif polarity < -0.1:
            reasons.append("mild negativity in your words made me more empathetic")
        elif polarity > 0.3:
            reasons.append("your positive energy made me more celebratory")

        # Memory influence
        if memory_context and len(memory_context) > 20:
            reasons.append("I'm drawing on our past conversations for context")

        # Wisdom / exercise triggers
        if has_wisdom:
            reasons.append("I chose a sacred quote that matched your emotional state")
        if has_exercise:
            reasons.append("I offered a breathing exercise because sustained "
                           "anxiety was detected")

        # Format
        explanation = "Why I responded this way: " + "; ".join(reasons) + "."
        return explanation

    def get_hud_text(self, max_lines: int = 3) -> list:
        """
        Return compact explanation lines suitable for the camera HUD overlay.
        """
        if not self._last_explanation:
            return ["No emotion data yet"]

        e = self._last_explanation
        lines = [
            f"XAI: {e['emotion'].upper()} ({e['confidence']:.0f}%) — {e['confidence_level']}",
            f"Cues: {', '.join(e['cues'][:2])}",
        ]
        if e.get("secondary_note"):
            lines.append(e["secondary_note"])
        return lines[:max_lines]

    def get_context_for_prompt(self) -> str:
        """
        Build an XAI context string for injection into the AI system prompt.
        This lets the AI know WHY the emotion was detected.
        """
        if not self._last_explanation:
            return ""

        e = self._last_explanation
        lines = [
            f"[XAI Context] Detected {e['emotion']} "
            f"({e['confidence']:.0f}% confidence, {e['confidence_level']}).",
            f"  Facial cues: {e['cue_text']}.",
            f"  Score breakdown: {e['breakdown']}.",
        ]
        if e.get("secondary_note"):
            lines.append(f"  {e['secondary_note']}.")
        return "\n".join(lines)

    def get_user_explanation(self) -> str:
        """
        Generate a user-facing explanation (shown in chat or TTS).
        Called when user asks 'why do you think I'm feeling X?'
        """
        if not self._last_explanation:
            return "I don't have enough data to explain yet. Let me look at you for a moment."

        e = self._last_explanation
        cues = " and ".join(e["cues"][:2])
        return (f"I noticed {cues} in your expression, which suggests "
                f"{e['emotion']}. My confidence is {e['confidence_level']} "
                f"at {e['confidence']:.0f}%. {e.get('secondary_note', '')}")

    def _empty_explanation(self, emotion: str) -> dict:
        return {
            "summary": f"Detected: {emotion} (no detailed scores available)",
            "cues": FACIAL_CUE_MAP.get(emotion, []),
            "cue_text": "No detailed cues available",
            "breakdown": f"{emotion}: —",
            "confidence_level": "unknown",
            "confidence_note": "Score data unavailable",
            "secondary_note": "",
            "emotion": emotion,
            "confidence": 0,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "raw_scores": {},
        }


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    xai = EmotionExplainer()

    print("=" * 55)
    print("  EchoMirror Explainable AI — Test Suite")
    print("=" * 55)

    # Test 1: Emotion explanation
    print("\n[TEST 1] Emotion Detection Explanation")
    scores = {
        "happy": 5.2, "sad": 15.5, "angry": 58.3,
        "fear": 8.1, "surprise": 2.9, "disgust": 3.8, "neutral": 6.2
    }
    result = xai.explain_detection("angry", scores, 58.3)
    print(f"  Summary   : {result['summary']}")
    print(f"  Breakdown : {result['breakdown']}")
    print(f"  Confidence: {result['confidence_level']} — {result['confidence_note']}")
    if result['secondary_note']:
        print(f"  Secondary : {result['secondary_note']}")

    # Test 2: Fusion explanation
    print("\n[TEST 2] Multimodal Fusion Explanation")
    fusion = xai.explain_fusion("sad", 72.0, "Negative", -0.45)
    print(f"  Summary   : {fusion['fusion_summary']}")
    print(f"  Agreement : {fusion['agreement']}")
    print(f"  Dominant  : {fusion['dominant_label']}")

    # Test 3: Conflict detection
    print("\n[TEST 3] Conflict Detection (face=happy, text=negative)")
    fusion2 = xai.explain_fusion("happy", 85.0, "Very Negative", -0.7)
    print(f"  Summary   : {fusion2['fusion_summary']}")
    print(f"  Conflict? : {fusion2['is_conflict']}")

    # Test 4: Response explanation
    print("\n[TEST 4] Response Explanation")
    resp_explain = xai.explain_response(
        "sad", "Negative", -0.45,
        memory_context="User has been sad for 3 days",
        has_wisdom=True, has_exercise=False
    )
    print(f"  {resp_explain}")

    # Test 5: HUD text
    print("\n[TEST 5] HUD Overlay Text")
    for line in xai.get_hud_text():
        print(f"  {line}")

    # Test 6: Prompt context
    print("\n[TEST 6] System Prompt XAI Context")
    print(xai.get_context_for_prompt())

    # Test 7: User-facing explanation
    print("\n[TEST 7] User-Facing Explanation")
    print(f"  {xai.get_user_explanation()}")

    print("\n[XAI] All tests passed! ✅")
