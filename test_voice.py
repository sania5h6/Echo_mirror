"""
EchoMirror - Find Indian Voice IDs from your ElevenLabs account
"""
import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import pygame
import io
import time

load_dotenv(override=True)

api_key = os.getenv("ELEVENLABS_API_KEY")
print(f"API Key: {api_key[:15]}...\n")

client = ElevenLabs(api_key=api_key)
pygame.mixer.init(frequency=44100)

TEST_TEXT = "Hello, I am EchoMirror. I am here for you, no judgment, no rush."

def test_voice(voice_id, voice_name, gender):
    try:
        print(f"  Testing {voice_name} ({gender})...", end=" ", flush=True)
        audio_gen = client.text_to_speech.convert(
            voice_id=voice_id,
            text=TEST_TEXT,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        audio_bytes = b"".join(audio_gen)
        sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))
        sound.play()
        time.sleep(sound.get_length() + 0.5)
        print(f"✅ WORKS → ID: {voice_id}")
        return True
    except Exception as e:
        err = str(e)
        if "402" in err or "payment" in err.lower():
            print("❌ Paid only")
        elif "401" in err:
            print("❌ Invalid API key")
        else:
            print(f"❌ {err[:80]}")
        return False

# First try to list all voices on your account
print("Fetching all voices on your account...\n")
try:
    all_voices = client.voices.get_all()
    print(f"Found {len(all_voices.voices)} voices:\n")
    indian_voices = []
    for v in all_voices.voices:
        print(f"  {v.name:30s} | {v.voice_id} | {v.category}")
        name_lower = v.name.lower()
        if any(n in name_lower for n in ["devi", "monika", "raju", "muskaan", "india", "hindi"]):
            indian_voices.append({"id": v.voice_id, "name": v.name, "gender": "unknown"})

    print(f"\nFound {len(indian_voices)} Indian voices: {[v['name'] for v in indian_voices]}\n")

    if indian_voices:
        print("Testing Indian voices...\n")
        working = []
        for v in indian_voices:
            ok = test_voice(v["id"], v["name"], v["gender"])
            if ok:
                working.append(v)

        print("\n" + "="*55)
        print("WORKING INDIAN VOICES:")
        for w in working:
            print(f"  {w['name']:25s} → {w['id']}")
        print("="*55)

except Exception as e:
    print(f"Could not list voices: {e}")
    print("\nFalling back to manual ID testing...\n")

    # Manual IDs for the voices you mentioned
    FREE_VOICES = [
    # Confirmed working
    {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella",          "gender": "female"},
    {"id": "ErXwobaYiN019PkySvjV", "name": "Antoni",         "gender": "male"},
    {"id": "VR6AewLTigWG4xSOukaG", "name": "Arnold",         "gender": "male"},
    {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam",           "gender": "male"},
    # Indian voices
    {"id": "RXe6OFmxoC0nlSWpuCDy", "name": "Anika (Indian)", "gender": "female"},
    {"id": "MF4J4IDTRo0AxOO4dpFR", "name": "Devi",           "gender": "female"},
]

    working = []
    for v in FREE_VOICES:
        ok = test_voice(v["id"], v["name"], v["gender"])
        if ok:
            working.append(v)

    print("\n" + "="*55)
    print("WORKING VOICES:")
    for w in working:
        print(f"  {w['name']:25s} → {w['id']}")
    print("="*55)