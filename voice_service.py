import os
import io
import re
import wave
from typing import List, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


class VoiceService:
    """
    Speech-to-Text and Text-to-Speech via the Gemini API.

    Uses the stable generateContent API rather than the beta Interactions API.
    Google recommends generateContent for production; the Interactions API is
    still preview and its schema may change. The TTS model and voices are the
    same either way — only the call shape differs.
    """

    STT_MODEL = "gemini-2.0-flash"
    TTS_MODEL = os.getenv("TTS_MODEL", "gemini-3.1-flash-tts-preview")

    # Aoede | Charon | Fenrir | Kore | Puck
    TTS_VOICE = os.getenv("TTS_VOICE", "Kore")

    SAMPLE_RATE = 24000
    SAMPLE_WIDTH = 2
    CHANNELS = 1

    # 24 kHz * 16-bit mono is 48 KB per second of audio, uncompressed. Vercel
    # caps responses at 4.5 MB, so a single request must stay well under
    # ~90 seconds of speech. Callers should chunk longer text.
    MAX_TTS_CHARS = int(os.getenv("MAX_TTS_CHARS", "700"))

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        self.client = genai.Client(api_key=api_key)

    # ── Speech-to-Text ────────────────────────────────────────────────────────

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        response = self.client.models.generate_content(
            model=self.STT_MODEL,
            contents=[
                (
                    "Transcribe the spoken words in this audio exactly as heard. "
                    "Return only the transcription — no labels, no punctuation changes, "
                    "no explanations."
                ),
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            ],
        )
        return (response.text or "").strip()

    # ── Text-to-Speech ────────────────────────────────────────────────────────

    @staticmethod
    def strip_markdown(text: str) -> str:
        text = re.sub(r"```.*?```", " ", text, flags=re.S)
        text = re.sub(r"`([^`]*)`", r"\1", text)
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"\*(.*?)\*", r"\1", text)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"([.!?][\"']?)\s*\n{2,}\s*", r"\1 ", text)
        text = re.sub(r"([^\s.!?])\s*\n{2,}\s*", r"\1. ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def split_for_speech(cls, text: str, limit: Optional[int] = None) -> List[str]:
        """Splits on sentence boundaries so each request stays under the response
        size ceiling and the listener hears audio sooner."""
        limit = limit or cls.MAX_TTS_CHARS
        clean = cls.strip_markdown(text)
        if not clean:
            return []

        guarded = re.sub(
            r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|approx|fig|vol|no)\.",
            lambda m: m.group(1) + "\x00", clean, flags=re.I,
        )
        # Python's re rejects a variable-width lookbehind, so the terminator is
        # captured and stitched back onto the sentence it belongs to.
        pieces = re.split(r"([.!?][\"']?)(?:\s+)", guarded)
        sentences = []
        for i in range(0, len(pieces), 2):
            body = pieces[i]
            terminator = pieces[i + 1] if i + 1 < len(pieces) else ""
            sentence = (body + terminator).replace("\x00", ".").strip()
            if sentence:
                sentences.append(sentence)

        chunks: List[str] = []
        buf = ""
        for sentence in sentences:
            while len(sentence) > limit:
                cut = sentence.rfind(" ", 0, limit)
                cut = cut if cut > limit * 0.6 else limit
                chunks.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if len(buf) + len(sentence) + 1 <= limit:
                buf = f"{buf} {sentence}".strip()
            else:
                if buf:
                    chunks.append(buf)
                buf = sentence
        if buf:
            chunks.append(buf)
        return chunks

    def synthesize(self, text: str) -> bytes:
        """Returns a playable WAV. Gemini emits raw 16-bit PCM at 24 kHz mono,
        which browsers will not play without a container."""
        clean = self.strip_markdown(text)[: self.MAX_TTS_CHARS]
        if not clean:
            raise ValueError("Nothing to speak after removing formatting.")

        response = self.client.models.generate_content(
            model=self.TTS_MODEL,
            contents=clean,
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

        try:
            pcm_data = response.candidates[0].content.parts[0].inline_data.data
        except (AttributeError, IndexError, TypeError):
            raise RuntimeError("Gemini returned no audio for this text.")

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.SAMPLE_WIDTH)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(pcm_data)
        return buf.getvalue()


_voice_service_instance: Optional[VoiceService] = None


def get_voice_service() -> VoiceService:
    global _voice_service_instance
    if _voice_service_instance is None:
        _voice_service_instance = VoiceService()
    return _voice_service_instance