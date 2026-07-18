"""
============================================================
ESP32-P4 AI Voice Assistant
AI Server - Whisper Speech-to-Text Service

Processes audio in memory using Faster-Whisper without 
writing temporary files to disk. Supports both standard WAV 
files and raw 16-bit PCM streams from the ESP32.
============================================================
"""

import time
import io
import logging
import numpy as np
from typing import Optional
from faster_whisper import WhisperModel

import config
from models.response_models import TranscribeResponse

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WhisperService")


class WhisperService:
    """
    Singleton service class for handling Speech-to-Text using Faster-Whisper.
    """
    _instance: Optional["WhisperService"] = None
    _model: Optional[WhisperModel] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WhisperService, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        """
        Loads the Whisper model into GPU/CPU memory based on config.py.
        Called automatically only once when the service is first initialized.
        """
        logger.info("===========================================")
        logger.info(f"Loading Faster-Whisper Model: '{config.WHISPER_MODEL}'")
        logger.info(f"Target Device: {config.WHISPER_DEVICE} ({config.WHISPER_COMPUTE_TYPE})")
        logger.info("===========================================")

        start_time = time.time()
        try:
            self._model = WhisperModel(
                model_size_or_path=config.WHISPER_MODEL,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE
            )
            load_time = round(time.time() - start_time, 2)
            logger.info(f"Whisper Model loaded successfully in {load_time}s!")
        except Exception as e:
            logger.error(f"Failed to load Whisper model on {config.WHISPER_DEVICE}: {e}")
            logger.warning("Attempting fallback to CPU (int8)...")
            self._model = WhisperModel(
                model_size_or_path=config.WHISPER_MODEL,
                device="cpu",
                compute_type="int8"
            )
            logger.info("Whisper Model loaded successfully on CPU fallback!")

    def transcribe_audio(self, audio_bytes: bytes, is_raw_pcm: bool = False) -> TranscribeResponse:
        """
        Transcribes audio bytes into text.

        :param audio_bytes: The raw byte payload received from the HTTP request.
        :param is_raw_pcm: If True, treats bytes as raw 16-bit 16kHz mono PCM (no WAV header).
                           If False, treats bytes as a standard WAV file container.
        :return: TranscribeResponse Pydantic model matching ESP32 cJSON expectations.
        """
        if not self._model:
            raise RuntimeError("Whisper model is not initialized.")

        start_time = time.time()
        logger.info(f"Starting transcription. Payload size: {len(audio_bytes)} bytes | Raw PCM: {is_raw_pcm}")

        try:
            if is_raw_pcm:
                # Convert raw 16-bit PCM bytes directly to a normalized float32 NumPy array (-1.0 to 1.0)
                # Faster-Whisper natively consumes 16kHz float32 arrays at lightning speed!
                int16_samples = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_input = int16_samples.astype(np.float32) / 32768.0
            else:
                # If sending a formatted WAV file from ESP32, wrap in an in-memory stream
                audio_input = io.BytesIO(audio_bytes)

            # Run transcription
            segments, info = self._model.transcribe(
                audio=audio_input,
                beam_size=5,
                language="en",
                vad_filter=True,  # Built-in Silero VAD to strip remaining background silence
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Combine all transcribed segments into a single clean string
            transcribed_text = " ".join([segment.text for segment in segments]).strip()
            
            processing_time_ms = round((time.time() - start_time) * 1000, 2)
            audio_duration = round(info.duration, 2)

            logger.info("========== TRANSCRIPTION SUCCESS ==========")
            logger.info(f"Recognized Text : \"{transcribed_text}\"")
            logger.info(f"Audio Duration  : {audio_duration}s")
            logger.info(f"Processing Time : {processing_time_ms}ms")
            logger.info("===========================================")

            return TranscribeResponse(
                text=transcribed_text,
                language=info.language,
                duration_seconds=audio_duration,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            logger.error(f"Error during audio transcription: {str(e)}")
            # Return an empty transcription string rather than crashing the pipeline
            return TranscribeResponse(
                text="",
                language="en",
                duration_seconds=0.0,
                processing_time_ms=round((time.time() - start_time) * 1000, 2)
            )

# Create an easily importable singleton instance
whisper_service = WhisperService()
