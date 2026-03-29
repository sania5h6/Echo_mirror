import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

load_dotenv()
print("API KEY:", os.getenv("ELEVENLABS_API_KEY"))
client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

audio = client.text_to_speech.convert(
    voice_id="EXAVITQu4vr4xnSDxMaL",
    model_id="eleven_multilingual_v2",
    text="Hello Sania, Echo Mirror is now ready"
)

play(audio)