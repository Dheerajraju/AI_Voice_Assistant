"""
============================================================
ESP32-P4 AI Voice Assistant
AI Server - DeepSeek LLM Service

Handles conversational AI responses using DeepSeek's 
OpenAI-compatible API endpoint. Automatically formats 
prompts to be clean and natural for Text-to-Speech (TTS).
============================================================
"""

import time
import logging
from typing import List, Optional
from openai import AsyncOpenAI, OpenAIError

import config
from models.response_models import ChatMessage, ChatResponse

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepSeekService")

# System prompt designed specifically for voice assistants (avoids weird TTS reading artifacts)
VOICE_ASSISTANT_SYSTEM_PROMPT = (
    "You are a helpful, intelligent, and friendly AI voice assistant running on an ESP32-P4 hardware device. "
    "Your responses will be read aloud by a Text-to-Speech (TTS) engine. "
    "Therefore, keep your answers concise, conversational, and direct. "
    "STRICTLY AVOID using Markdown formatting, asterisks, bullet points, numbered lists, emojis, URLs, or code blocks, "
    "as they sound unnatural when synthesized into spoken audio."
)


class DeepSeekService:
    """
    Singleton service class for communicating with the DeepSeek API.
    """
    _instance: Optional["DeepSeekService"] = None
    _client: Optional[AsyncOpenAI] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeepSeekService, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self):
        """
        Initializes the async OpenAI client pointing to DeepSeek's base URL.
        """
        logger.info("===========================================")
        logger.info("Initializing DeepSeek LLM Client...")
        logger.info(f"Target Model : '{config.DEEPSEEK_MODEL}'")
        logger.info("===========================================")

        if not config.DEEPSEEK_API_KEY or config.DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY":
            logger.warning("⚠️ DEEPSEEK_API_KEY is not set in config.py or environment variables!")

        try:
            # DeepSeek uses an OpenAI-compatible endpoint structure
            self._client = AsyncOpenAI(
                api_key=config.DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com"
            )
            logger.info("DeepSeek client initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek client: {e}")

    async def generate_reply(
        self, 
        message: str, 
        history: Optional[List[ChatMessage]] = None
    ) -> ChatResponse:
        """
        Sends the user's transcribed speech to DeepSeek and returns the conversational reply.

        :param message: The latest text message/prompt from the user.
        :param history: Optional list of previous ChatMessage objects for multi-turn context.
        :return: ChatResponse Pydantic model matching ESP32 cJSON expectations.
        """
        if not self._client:
            raise RuntimeError("DeepSeek client is not initialized.")

        start_time = time.time()
        logger.info(f"Sending prompt to DeepSeek: \"{message}\"")

        # 1. Build message payload starting with the system instructions
        messages_payload = [
            {"role": "system", "content": VOICE_ASSISTANT_SYSTEM_PROMPT}
        ]

        # 2. Append prior conversation history if provided
        if history:
            for turn in history:
                messages_payload.append({"role": turn.role, "content": turn.content})

        # 3. Append the latest user prompt
        messages_payload.append({"role": "user", "content": message})

        try:
            # Call DeepSeek API asynchronously
            response = await self._client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=messages_payload,
                temperature=0.7,
                max_tokens=250,  # Keep responses relatively brief for comfortable voice playback
                top_p=0.95,
                stream=False
            )

            # Extract the generated text
            ai_reply_text = response.choices[0].message.content.strip()
            processing_time_ms = round((time.time() - start_time) * 1000, 2)

            logger.info("========== DEEPSEEK RESPONSE ==========")
            logger.info(f"AI Reply        : \"{ai_reply_text}\"")
            logger.info(f"Tokens Used     : {response.usage.total_tokens}")
            logger.info(f"Processing Time : {processing_time_ms}ms")
            logger.info("=======================================")

            return ChatResponse(
                response=ai_reply_text,
                model_used=config.DEEPSEEK_MODEL,
                processing_time_ms=processing_time_ms
            )

        except OpenAIError as api_err:
            logger.error(f"DeepSeek API Error: {str(api_err)}")
            processing_time_ms = round((time.time() - start_time) * 1000, 2)
            
            # Return a graceful spoken fallback so the ESP32 speaker doesn't stay silent
            fallback_message = "I'm sorry, I am having trouble connecting to my cloud AI servers right now. Please try again in a moment."
            return ChatResponse(
                response=fallback_message,
                model_used=config.DEEPSEEK_MODEL,
                processing_time_ms=processing_time_ms
            )
        except Exception as e:
            logger.error(f"Unexpected error during LLM generation: {str(e)}")
            return ChatResponse(
                response="I encountered an unexpected system error while processing your request.",
                model_used=config.DEEPSEEK_MODEL,
                processing_time_ms=round((time.time() - start_time) * 1000, 2)
            )

# Create an easily importable singleton instance
deepseek_service = DeepSeekService()
