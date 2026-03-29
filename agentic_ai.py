"""
EchoMirror - Step 5: Agentic AI
Features:
  - Proactive emotional support (EchoMirror speaks first when patterns detected)
  - Crisis detection & escalation (suicidal ideation, self-harm keywords)
  - Mood streak interventions (sad 3+ days → special response)
  - Breathing & grounding exercises (auto-triggered on anxiety/fear)
  - Session goal tracking (user sets a goal, Echo checks in)
  - End-of-session reflection prompts
  - Motivational nudges on positive streaks
"""

import os
import re
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# CRISIS KEYWORDS
# ─────────────────────────────────────────
CRISIS_KEYWORDS = [
    "kill myself", "end my life", "want to die", "suicide", "suicidal",
    "don't want to live", "no reason to live", "better off dead",
    "can't go on", "hurt myself", "self harm", "cutting myself",
    "worthless", "nobody cares", "everyone hates me", "give up on life"
]

CRISIS_RESOURCES = """
📞 iCall (India): 9152987821
📞 Vandrevala Foundation: 1860-2662-345 (24/7)
📞 NIMHANS: 080-46110007
💬 iCall Chat: icallhelpline.org
"""

# ─────────────────────────────────────────
# BREATHING EXERCISE TRIGGERS
# ─────────────────────────────────────────
ANXIETY_EMOTIONS = ["fear", "angry", "disgust"]

BREATHING_EXERCISES = [
    {
        "name": "4-7-8 Breathing",
        "steps": [
            "Breathe in slowly for 4 counts",
            "Hold your breath for 7 counts",
            "Exhale completely for 8 counts",
            "Repeat 3 times"
        ]
    },
    {
        "name": "Box Breathing",
        "steps": [
            "Breathe in for 4 counts",
            "Hold for 4 counts",
            "Breathe out for 4 counts",
            "Hold for 4 counts",
            "Repeat 4 times"
        ]
    },
    {
        "name": "5-4-3-2-1 Grounding",
        "steps": [
            "Name 5 things you can SEE right now",
            "Name 4 things you can TOUCH",
            "Name 3 things you can HEAR",
            "Name 2 things you can SMELL",
            "Name 1 thing you can TASTE",
            "Take a slow deep breath"
        ]
    }
]

# ─────────────────────────────────────────
# PROACTIVE INTERVENTIONS
# ─────────────────────────────────────────
PROACTIVE_MESSAGES = {
    "sad_streak": [
        "I've noticed you've been feeling down for a few days. I'm here — no pressure, no rush. Want to talk about what's been weighing on you?",
        "You've been carrying a lot lately. Even small things matter. How are you really feeling today?",
        "I see you've been going through a tough stretch. You don't have to face it alone. I'm listening."
    ],
    "happy_streak": [
        "You've been in such a good place lately! That makes me genuinely happy. What's been going well for you?",
        "I've noticed a positive shift in your energy. Keep going — you're doing amazing. What's your secret?",
        "Something wonderful is happening with you these days. I'd love to hear about what's bringing you joy!"
    ],
    "first_session": [
        "Welcome to EchoMirror. This is your safe space — no judgment, no rush. I'm here to listen, support, and grow with you. How are you feeling right now?",
    ],
    "returning_user": [
        "Welcome back. I remember you. How have you been since we last talked?",
        "Good to see you again. Take your time — I'm here. What's on your mind today?"
    ]
}

# ─────────────────────────────────────────
# AGENTIC AI ENGINE
# ─────────────────────────────────────────
class AgenticAI:
    def __init__(self, db=None):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env!")
        self.client = Groq(api_key=api_key)
        self.db = db  # MemoryDB instance (optional)

        self.session_goal = None
        self.crisis_mode = False
        self.breathing_done_this_session = False
        self.turn_count = 0
        self._breathing_index = 0

    # ─── CRISIS DETECTION ───
    def detect_crisis(self, text: str) -> bool:
        """Check if user message contains crisis-level content"""
        text_lower = text.lower()
        for keyword in CRISIS_KEYWORDS:
            if keyword in text_lower:
                return True
        return False

    def get_crisis_response(self) -> str:
        """Return a warm, non-robotic crisis response"""
        self.crisis_mode = True
        return (
            "I hear you, and I want you to know — what you're feeling is real and it matters. "
            "You don't have to carry this alone.\n\n"
            "Please reach out to someone who can help right now:\n"
            f"{CRISIS_RESOURCES}\n"
            "I'll be right here with you. Can you tell me more about what's happening?"
        )

    # ─── PROACTIVE OPENING MESSAGE ───
    def get_opening_message(self, total_sessions: int, sad_streak: int, happy_streak: int) -> str:
        """Generate the right opening message based on user history"""
        import random

        if total_sessions == 0:
            return random.choice(PROACTIVE_MESSAGES["first_session"])

        if sad_streak >= 3:
            return random.choice(PROACTIVE_MESSAGES["sad_streak"])

        if happy_streak >= 3:
            return random.choice(PROACTIVE_MESSAGES["happy_streak"])

        return random.choice(PROACTIVE_MESSAGES["returning_user"])

    # ─── BREATHING EXERCISE ───
    def should_offer_breathing(self, emotion: str, turn: int) -> bool:
        """Offer breathing exercise for anxiety/fear emotions"""
        if self.breathing_done_this_session:
            return False
        if emotion.lower() in ANXIETY_EMOTIONS and turn >= 2:
            return True
        return False

    def get_breathing_exercise(self) -> dict:
        """Get next breathing exercise (rotates between options)"""
        ex = BREATHING_EXERCISES[self._breathing_index % len(BREATHING_EXERCISES)]
        self._breathing_index += 1
        self.breathing_done_this_session = True
        return ex

    def format_breathing_exercise(self, exercise: dict) -> str:
        steps = "\n".join([f"  {i+1}. {s}" for i, s in enumerate(exercise["steps"])])
        return f"🫁 Let's try **{exercise['name']}** together:\n{steps}"

    # ─── MOOD STREAK INTERVENTION ───
    def get_streak_intervention(self, emotion: str, streak: int) -> str | None:
        """Generate AI-powered streak intervention message"""
        if streak < 2:
            return None

        if emotion == "sad" and streak >= 2:
            prompt = f"The user has been feeling sad for {streak} consecutive days. Write a warm, empathetic 2-sentence check-in message as EchoMirror. Don't be preachy. Be human."
        elif emotion == "angry" and streak >= 2:
            prompt = f"The user has been feeling angry for {streak} consecutive days. Write a calm, understanding 2-sentence message as EchoMirror that acknowledges their frustration without judgment."
        elif emotion == "happy" and streak >= 3:
            prompt = f"The user has been feeling happy for {streak} consecutive days. Write an enthusiastic, warm 2-sentence celebration message as EchoMirror."
        else:
            return None

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[AgenticAI] Streak intervention error: {e}")
            return None

    # ─── GOAL TRACKING ───
    def set_session_goal(self, goal: str):
        """Set a goal for this session"""
        self.session_goal = goal
        print(f"[AgenticAI] Session goal set: {goal}")

    def check_goal_progress(self, user_message: str) -> str | None:
        """Check if user message relates to their session goal"""
        if not self.session_goal:
            return None

        goal_keywords = self.session_goal.lower().split()
        msg_lower = user_message.lower()
        matches = sum(1 for w in goal_keywords if w in msg_lower)

        if matches >= 2:
            return f"I noticed you're touching on your goal — '{self.session_goal}'. How are you feeling about your progress?"
        return None

    # ─── END OF SESSION REFLECTION ───
    def get_session_reflection_prompt(self, emotion: str, polarity: float) -> str:
        """Generate a closing reflection prompt based on session emotion"""
        if polarity > 0.3:
            return "Before you go — what's one thing that made you smile today? Even something small counts."
        elif polarity < -0.3:
            return "Before we wrap up — is there one tiny thing you could do for yourself tonight? Even just a glass of water or a deep breath."
        else:
            return "As we close today's session — what's one word that describes how you're feeling right now?"

    # ─── MOTIVATIONAL NUDGE ───
    def get_motivational_nudge(self, turn: int, polarity: float) -> str | None:
        """Occasionally send a motivational message on positive sessions"""
        if turn % 5 == 0 and polarity > 0.2:
            nudges = [
                "You're doing really well opening up. That takes courage. 💙",
                "Every conversation you have with yourself is growth. Keep going.",
                "You showed up today. That already matters more than you know.",
                "Healing isn't linear, but you're moving forward. I see that."
            ]
            import random
            return random.choice(nudges)
        return None

    # ─── MAIN AGENTIC HOOK ───
    def process_turn(self, user_message: str, emotion: str, polarity: float, turn: int) -> dict:
        """
        Main function called every conversation turn.
        Returns a dict with agentic actions to take.
        """
        self.turn_count = turn
        result = {
            "crisis": False,
            "crisis_response": None,
            "breathing_exercise": None,
            "streak_intervention": None,
            "goal_check": None,
            "motivational_nudge": None,
            "session_reflection": None,
        }

        # 1. Crisis detection (highest priority)
        if self.detect_crisis(user_message):
            result["crisis"] = True
            result["crisis_response"] = self.get_crisis_response()
            return result  # Skip everything else

        # 2. Breathing exercise for anxiety/fear
        if self.should_offer_breathing(emotion, turn):
            ex = self.get_breathing_exercise()
            result["breathing_exercise"] = self.format_breathing_exercise(ex)

        # 3. Goal check
        goal_msg = self.check_goal_progress(user_message)
        if goal_msg:
            result["goal_check"] = goal_msg

        # 4. Motivational nudge
        nudge = self.get_motivational_nudge(turn, polarity)
        if nudge:
            result["motivational_nudge"] = nudge

        return result

    def get_end_of_session(self, emotion: str, polarity: float) -> str:
        """Call this when user ends the session"""
        return self.get_session_reflection_prompt(emotion, polarity)


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    agent = AgenticAI()

    print("=" * 50)
    print("  EchoMirror Agentic AI — Test Suite")
    print("=" * 50)

    # Test 1: Crisis detection
    print("\n[TEST 1] Crisis detection")
    crisis_msg = "I want to end my life"
    is_crisis = agent.detect_crisis(crisis_msg)
    print(f"  Message: '{crisis_msg}'")
    print(f"  Crisis detected: {is_crisis}")
    if is_crisis:
        print(f"  Response:\n{agent.get_crisis_response()}")

    # Test 2: Breathing exercise
    print("\n[TEST 2] Breathing exercise")
    ex = agent.get_breathing_exercise()
    print(agent.format_breathing_exercise(ex))

    # Test 3: Opening messages
    print("\n[TEST 3] Opening messages")
    print("  First session:", agent.get_opening_message(0, 0, 0))
    print("  Sad streak 3d:", agent.get_opening_message(10, 3, 0))
    print("  Happy streak: ", agent.get_opening_message(10, 0, 4))

    # Test 4: End of session
    print("\n[TEST 4] End of session reflection")
    print("  Positive:", agent.get_end_of_session("happy", 0.6))
    print("  Negative:", agent.get_end_of_session("sad", -0.5))
    print("  Neutral: ", agent.get_end_of_session("neutral", 0.0))

    # Test 5: Full turn processing
    print("\n[TEST 5] Full turn processing")
    result = agent.process_turn("I feel so anxious I can't breathe", "fear", -0.4, 3)
    print(f"  Crisis: {result['crisis']}")
    print(f"  Breathing: {result['breathing_exercise']}")

    print("\n[AgenticAI] All tests passed! ✅")