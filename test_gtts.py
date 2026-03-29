"""
EchoMirror - gTTS Voice Test
Tests Google TTS with Indian English accent
"""
import os
from gtts import gTTS
import pygame
import io
import time

pygame.mixer.init(frequency=44100)

def speak(text, lang='en', tld='co.in'):
    """
    lang = 'en' (English)
    tld options for accent:
      'co.in' = Indian English
      'com'   = American English
      'co.uk' = British English
      'com.au'= Australian English
    """
    tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    pygame.mixer.music.load(mp3_fp)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

print("Testing gTTS voices...\n")

test_text = "Hello, I am EchoMirror. I am here to listen and support you today."

print("1. Indian English (co.in)...")
speak(test_text, tld='co.in')
time.sleep(0.5)

print("2. American English (com)...")
speak(test_text, tld='com')
time.sleep(0.5)

print("3. British English (co.uk)...")
speak(test_text, tld='co.uk')
time.sleep(0.5)

print("\nNow testing emotional phrases...\n")

phrases = [
    "It sounds like you are carrying a heavy weight on your heart right now.",
    "You are not alone in this. I am right here with you.",
    "Even the darkest night will end and the sun will rise.",
    "Take a deep breath. You have got this.",
]

for p in phrases:
    print(f"Speaking: {p[:50]}...")
    speak(p, tld='co.in')
    time.sleep(0.3)

print("\nDone! Which accent did you like best?")
print("co.in = Indian | com = American | co.uk = British")