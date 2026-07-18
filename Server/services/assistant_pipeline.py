"""
============================================================
ESP32-P4 AI Voice Assistant
AI Server - Unified Assistant Pipeline Coordinator

Orchestrates the complete end-to-end flow in memory:
Audio Bytes (ESP32) -> Whisper (STT) -> DeepSeek (LLM) -> Kokoro (TTS)
============================================================
"""

import time
import logging
from typing import AsyncGenerator, List, Optional

from models.response_models import ChatMessage, AssistantResponse
from services.whisper_service import whisper_service
from services.deepseek_service import deepseek_service
from services.kokoro_service import kokoro_service

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AssistantPipeline")


class AssistantPipeline:
    """
    Singleton class coordinating the full Speech-to-Text -> LLM -> Text-to-Speech
    processing loop.
    """
    _instance: Optional["AssistantPipeline"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AssistantPipeline, cls).__new__(cls)
        return cls._instance

    async def execute_full_pipeline(
        self, 
        audio_bytes: bytes, 
        is_raw_pcm: bool = False,
        history: Optional[List[ChatMessage]] = None
    ) -> AssistantResponse:
        """
        Executes a complete standard execution loop: 
        Transcribes audio, calls the LLM, and returns a static structured response.
        
        :param audio_bytes: Raw incoming audio file/samples payload.
        :param is_raw_pcm: True if raw 16kHz mono PCM, False if structured WAV file.
        :param history: Prior conversation turns for maintaining context.
        :return: AssistantResponse containing text results.
        """
        start_time = time.time()
        logger.info("⚡ Executing Full Text-Based Assistant Pipeline Turn...")

        # 1. Step 1: Speech-to-Text (Whisper)
        stt_result = whisper_service.transcribe_audio(audio_bytes, is_raw_pcm=is_raw_pcm)
        
        if not stt_result.text or stt_result.text.strip() == "":
            logger.warning("Pipeline aborted: No speech text was identified in the audio.")
            return AssistantResponse(
                status="error",
                recognized_text="",
                ai_reply="I didn't catch that. Could you please repeat yourself?",
                total_latency_ms=round((time.time() - start_time) * 1000, 2)
            )

        # 2. Step 2: Language Model Reasoning (DeepSeek)
        llm_result = await deepseek_service.generate_reply(stt_result.text, history=history)

        total_latency = round((time.time() - start_time) * 1000, 2)
        logger.info(f"✨ Standard Turn execution complete. Latency: {total_latency}ms")

        return AssistantResponse(
            status="success",
            recognized_text=stt_result.text,
            ai_reply=llm_result.response,
            total_latency_ms=total_latency
        )

    async def stream_assistant_audio(
        self, 
        audio_bytes: bytes, 
        is_raw_pcm: bool = False,
        history: Optional[List[ChatMessage]] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Low-Latency Streaming Pipeline.
        1. Fully transcribes incoming audio and queries DeepSeek.
        2. Immediately streams back raw 16kHz PCM audio bytes chunk-by-chunk 
           as Kokoro completes sentences, giving a real-time responsive voice.

        :param audio_bytes: Incoming audio sample array/file.
        :param is_raw_pcm: True if raw PCM, False if standard WAV container.
        :param history: Prior context frames.
        :yields: Sequential raw 16-bit 16kHz Mono PCM blocks.
        """
        logger.info("⚡ Executing Low-Latency Streaming Audio Pipeline Turn...")
        
        # 1. Step 1: Speech-to-Text
        stt_result = whisper_service.transcribe_audio(audio_bytes, is_raw_pcm=is_raw_pcm)
        if not stt_result.text or stt_result.text.strip() == "":
            logger.warning("Streaming Pipeline aborted: Speech was silent.")
            # Yield an error announcement chunk if synthesized or exit cleanly
            error_pcm, _ = kokoro_service.generate_pcm("I couldn't hear you clearly.")
            yield error_pcm
            return

        # 2. Step 2: Query DeepSeek for response string
        llm_result = await deepseek_service.generate_reply(stt_result.text, history=history)

        # 3. Step 3: Sentence-by-sentence TTS stream directly into the network pipe
        # Your ESP32-P4 speaker will start talking while the later sentences synthesize!
        for pcm_chunk in kokoro_service.stream_pcm(llm_result.response):
            if pcm_chunk:
                yield pcm_chunk


# Create an easily importable singleton instance
assistant_pipeline = AssistantPipeline()
