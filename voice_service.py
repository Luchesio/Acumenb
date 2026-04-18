import os
import io
import wave
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


class VoiceService:
    """
    Handles Speech-to-Text and Text-to-Speech via the Gemini API.

    Uses the newer `google-genai` SDK (genai.Client) which supports
    inline audio input and the TTS response modality.
    """

    # STT: any Gemini model that accepts audio content
    STT_MODEL = "gemini-2.0-flash"

    # TTS: dedicated TTS preview model
    TTS_MODEL = "gemini-3.1-flash-tts-preview"

    # Voice used for TTS — swap to any prebuilt Gemini voice you prefer:
    # Aoede | Charon | Fenrir | Kore | Puck
    TTS_VOICE = "Kore"

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        self.client = genai.Client(api_key=api_key)

    # ── Speech-to-Text ────────────────────────────────────────────────────────

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        """
        Transcribe raw audio bytes to text.

        Supported mime_types that Gemini accepts:
          audio/webm  (Chrome MediaRecorder default)
          audio/ogg   (Firefox MediaRecorder default)
          audio/mp4
          audio/mp3 / audio/mpeg
          audio/wav
        """
        response = self.client.models.generate_content(
            model=self.STT_MODEL,
            contents=[
                (
                    "Transcribe the spoken words in this audio exactly as heard. "
                    "Return only the transcription — no labels, no punctuation changes, "
                    "no explanations."
                ),
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mime_type,
                ),
            ],
        )
        return response.text.strip()

    # ── Text-to-Speech ────────────────────────────────────────────────────────

    def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech and return a valid WAV file as bytes.

        Gemini TTS returns raw 16-bit PCM at 24 kHz mono.
        We wrap it in a WAV container so browsers can play it directly.
        """
        response = self.client.models.generate_content(
            model=self.TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.TTS_VOICE,
                        )
                    )
                ),
            ),
        )

        # Raw PCM bytes from Gemini
        pcm_data: bytes = response.candidates[0].content.parts[0].inline_data.data

        # Wrap PCM → WAV in memory (no disk I/O)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)       # mono
            wf.setsampwidth(2)       # 16-bit
            wf.setframerate(24000)   # 24 kHz
            wf.writeframes(pcm_data)
        return buf.getvalue()


# ── Singleton ─────────────────────────────────────────────────────────────────

_voice_service_instance: VoiceService | None = None


def get_voice_service() -> VoiceService:
    global _voice_service_instance
    if _voice_service_instance is None:
        _voice_service_instance = VoiceService()
    return _voice_service_instance