"""
EchoMirror - Sacred Wisdom Module
Draws healing quotes from Quran, Bible, Bhagavad Gita,
Buddha Dhammapada, and Stoic texts.
AI picks the most relevant quote based on user's emotional state.
"""

import os
import random
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# SACRED QUOTES DATABASE
# ─────────────────────────────────────────
SACRED_QUOTES = {

    "sad": [
        {"text": "Verily, with hardship comes ease.", "source": "Quran 94:5"},
        {"text": "The Lord is close to the brokenhearted and saves those who are crushed in spirit.", "source": "Bible, Psalm 34:18"},
        {"text": "You have the right to perform your duties, but you are not entitled to the fruits of your actions.", "source": "Bhagavad Gita 2:47"},
        {"text": "Even the darkest night will end and the sun will rise.", "source": "Buddha Dhammapada"},
        {"text": "The impediment to action advances action. What stands in the way becomes the way.", "source": "Marcus Aurelius"},
        {"text": "Do not grieve over what has passed, and do not worry about what has not yet come.", "source": "Quran, inspired"},
        {"text": "He heals the brokenhearted and binds up their wounds.", "source": "Bible, Psalm 147:3"},
        {"text": "Pain is certain, suffering is optional.", "source": "Buddha"},
        {"text": "Never be sad about what is lost. Rather be happy for what you still have.", "source": "Bhagavad Gita, inspired"},
    ],

    "angry": [
        {"text": "And do not let the hatred of a people prevent you from being just.", "source": "Quran 5:8"},
        {"text": "A gentle answer turns away wrath, but a harsh word stirs up anger.", "source": "Bible, Proverbs 15:1"},
        {"text": "Let a man overcome anger by non-anger, let him overcome evil by good.", "source": "Buddha Dhammapada 1:17"},
        {"text": "You will not be punished for your anger, you will be punished by your anger.", "source": "Buddha"},
        {"text": "The strong man is not one who is good at wrestling, but the one who controls himself when angry.", "source": "Prophet Muhammad (PBUH)"},
        {"text": "When you are offended at anyone's fault, turn to yourself and study your own failings.", "source": "Epictetus"},
        {"text": "Holding on to anger is like grasping a hot coal with the intent of throwing it at someone else.", "source": "Buddha"},
    ],

    "fear": [
        {"text": "Allah does not burden a soul beyond that it can bear.", "source": "Quran 2:286"},
        {"text": "Do not be afraid, for I am with you. Do not be discouraged, for I am your God.", "source": "Bible, Isaiah 41:10"},
        {"text": "Fear not. What is not real never was and never will be. What is real always was.", "source": "Bhagavad Gita 2:16"},
        {"text": "The whole secret of existence is to have no fear.", "source": "Buddha"},
        {"text": "He who has overcome his fears will truly be free.", "source": "Aristotle"},
        {"text": "You have power over your mind, not outside events. Realize this and you will find strength.", "source": "Marcus Aurelius"},
        {"text": "Courage is not the absence of fear, but the triumph over it.", "source": "Inspired"},
    ],

    "neutral": [
        {"text": "Indeed, Allah will not change the condition of a people until they change what is in themselves.", "source": "Quran 13:11"},
        {"text": "Be still and know that I am God.", "source": "Bible, Psalm 46:10"},
        {"text": "The soul is never born nor dies at any time. It is unborn, eternal, ever-existing and primeval.", "source": "Bhagavad Gita 2:20"},
        {"text": "Peace comes from within. Do not seek it without.", "source": "Buddha"},
        {"text": "Waste no more time arguing about what a good man should be. Be one.", "source": "Marcus Aurelius"},
        {"text": "The present moment is the only moment available to us.", "source": "Thich Nhat Hanh"},
    ],

    "happy": [
        {"text": "And He found you lost and guided you.", "source": "Quran 93:7"},
        {"text": "This is the day that the Lord has made; let us rejoice and be glad in it.", "source": "Bible, Psalm 118:24"},
        {"text": "When meditation is mastered, the mind is unwavering like the flame of a lamp in a windless place.", "source": "Bhagavad Gita 6:19"},
        {"text": "Thousands of candles can be lighted from a single candle. Happiness never decreases by being shared.", "source": "Buddha"},
        {"text": "Very little is needed to make a happy life; it is all within yourself.", "source": "Marcus Aurelius"},
        {"text": "Gratitude is the memory of the heart.", "source": "Jean-Baptiste Massieu"},
    ],

    "disgust": [
        {"text": "Speak good words or remain silent.", "source": "Prophet Muhammad (PBUH)"},
        {"text": "Do not repay anyone evil for evil.", "source": "Bible, Romans 12:17"},
        {"text": "One who is not disturbed by the incessant flow of desires can alone achieve peace.", "source": "Bhagavad Gita 2:70"},
        {"text": "In separateness lies the world's great misery, in compassion lies the world's true strength.", "source": "Buddha"},
        {"text": "If it is not right, do not do it; if it is not true, do not say it.", "source": "Marcus Aurelius"},
    ],

    "surprise": [
        {"text": "And He gives you from all you ask of Him.", "source": "Quran 14:34"},
        {"text": "For I know the plans I have for you, plans to prosper you and not to harm you.", "source": "Bible, Jeremiah 29:11"},
        {"text": "There are no accidents. Everything happens for a reason.", "source": "Bhagavad Gita, inspired"},
        {"text": "The mind is everything. What you think you become.", "source": "Buddha"},
        {"text": "Confine yourself to the present.", "source": "Marcus Aurelius"},
    ],

    # Special states
    "depression": [
        {"text": "Verily, with hardship comes ease. Verily, with hardship comes ease.", "source": "Quran 94:5-6"},
        {"text": "Come to me, all you who are weary and burdened, and I will give you rest.", "source": "Bible, Matthew 11:28"},
        {"text": "You are never alone. The divine is always with you, within you.", "source": "Bhagavad Gita, inspired"},
        {"text": "You yourself, as much as anybody in the entire universe, deserve your love and affection.", "source": "Buddha"},
        {"text": "The obstacle is the way. What blocks you becomes your path.", "source": "Marcus Aurelius"},
        {"text": "After every difficulty, Allah has promised ease. Hold on.", "source": "Quran, inspired"},
        {"text": "Even the darkest night will end and the sun will rise.", "source": "Victor Hugo / Dhammapada"},
    ],

    "lost": [
        {"text": "And whoever relies upon Allah — then He is sufficient for him.", "source": "Quran 65:3"},
        {"text": "Trust in the Lord with all your heart and lean not on your own understanding.", "source": "Bible, Proverbs 3:5"},
        {"text": "Set your heart on doing good. Do it over and over again, and you will be filled with joy.", "source": "Buddha Dhammapada"},
        {"text": "You are what your deep, driving desire is. As your desire is, so is your will.", "source": "Brihadaranyaka Upanishad"},
        {"text": "It is not death that a man should fear, but he should fear never beginning to live.", "source": "Marcus Aurelius"},
    ],
}


# ─────────────────────────────────────────
# AI-POWERED QUOTE SELECTOR
# ─────────────────────────────────────────
class SacredWisdom:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key) if api_key else None

    def get_quote_ai(self, emotion: str, user_message: str, sentiment: str) -> dict:
        """
        Uses Groq AI to pick the most relevant quote from the database
        based on the user's exact words and emotional state.
        Falls back to rule-based if API fails.
        """
        if not self.client:
            return self.get_quote_simple(emotion)

        # Build quote list for AI to choose from
        candidate_emotions = self._get_candidate_emotions(emotion, sentiment, user_message)
        candidates = []
        for emo in candidate_emotions:
            candidates.extend(SACRED_QUOTES.get(emo, []))

        if not candidates:
            candidates = SACRED_QUOTES.get("neutral", [])

        # Format candidates for AI
        quote_list = "\n".join([
            f"{i+1}. \"{q['text']}\" — {q['source']}"
            for i, q in enumerate(candidates)
        ])

        prompt = f"""A user is feeling {emotion} with {sentiment} sentiment.
They said: "{user_message}"

From these sacred quotes, pick the ONE most healing and relevant quote for this exact moment.
Reply with ONLY the number of the quote. Nothing else.

{quote_list}"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.3,
            )
            choice = response.choices[0].message.content.strip()
            idx = int(''.join(filter(str.isdigit, choice))) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except Exception as e:
            print(f"[SacredWisdom] AI selection failed, using fallback: {e}")

        return self.get_quote_simple(emotion)

    def get_quote_simple(self, emotion: str) -> dict:
        """Rule-based fallback — picks random quote for emotion"""
        quotes = SACRED_QUOTES.get(emotion, SACRED_QUOTES["neutral"])
        return random.choice(quotes)

    def _get_candidate_emotions(self, emotion: str, sentiment: str, message: str) -> list:
        """Determine which emotion categories to pull quotes from"""
        candidates = [emotion]

        # Add special categories based on keywords
        msg_lower = message.lower()
        if any(w in msg_lower for w in ["lost", "purpose", "meaning", "why", "direction"]):
            candidates.append("lost")
        if any(w in msg_lower for w in ["depressed", "hopeless", "empty", "worthless", "give up"]):
            candidates.append("depression")
        if sentiment in ["Negative", "Very Negative"]:
            candidates.append("sad")

        return list(dict.fromkeys(candidates))  # Remove duplicates, preserve order

    def format_quote(self, quote: dict) -> str:
        """Format for display"""
        return f'"{quote["text"]}"\n— {quote["source"]}'

    def get_healing_message(self, emotion: str, user_message: str,
                             sentiment: str, polarity: float) -> str:
        """
        Full healing response:
        Quote + brief contextual framing from AI
        """
        if not self.client:
            quote = self.get_quote_simple(emotion)
            return f"A moment of wisdom for you:\n\n{self.format_quote(quote)}"

        quote = self.get_quote_ai(emotion, user_message, sentiment)

        # Ask AI to frame the quote personally
        prompt = f"""The user is feeling {emotion} ({sentiment}, polarity {polarity:+.2f}).
They said: "{user_message}"

You selected this sacred quote for them:
"{quote['text']}" — {quote['source']}

Write 1-2 warm, personal sentences connecting this quote to what they shared.
Do not repeat the quote. Do not be preachy. Speak like a gentle friend.
Keep it under 40 words."""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.7,
            )
            framing = response.choices[0].message.content.strip()
            return f"{framing}\n\n✨ {self.format_quote(quote)}"
        except:
            return f"A moment of wisdom for you:\n\n✨ {self.format_quote(quote)}"


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    wisdom = SacredWisdom()

    test_cases = [
        ("sad",     "I feel really lost and don't know what to do",    "Negative",  -0.6),
        ("angry",   "I'm so frustrated with everything around me",      "Negative",  -0.7),
        ("fear",    "I'm scared about my future",                       "Negative",  -0.5),
        ("neutral", "I don't feel anything today just blank",           "Neutral",    0.0),
        ("happy",   "I finally achieved something I worked hard for",   "Positive",  +0.8),
    ]

    print("[SacredWisdom] Testing AI quote selection...\n")
    for emotion, message, sentiment, polarity in test_cases:
        print(f"Emotion: {emotion.upper()} | '{message[:45]}...'")
        result = wisdom.get_healing_message(emotion, message, sentiment, polarity)
        print(f"{result}")
        print("-" * 60)