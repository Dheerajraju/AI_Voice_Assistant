"""
============================================================
ESP32-P4 AI Voice Assistant
AI Server - FastAPI Main Entry Point

Binds individual processing engines into clean HTTP endpoints.
Supports standard multipart files and chunked streaming binary 
responses for real-time low-latency voice playback.
============================================================
"""

import time
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import config
from models.response_models import TranscribeResponse, AssistantResponse, ErrorResponse
from services.whisper_service import whisper_service
from services.deepseek_service import deepseek_service
from services.kokoro_service import kokoro_service
from services.assistant_pipeline import assistant_pipeline

# Initialize Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MAIN_SERVER")

# Initialize FastAPI App
app = FastAPI(
    title="ESP32-P4 AI Voice Assistant Backend",
    description="High-performance, memory-only Whisper STT, DeepSeek LLM, and Kokoro TTS Pipeline",
    version="2.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing) for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 1. Root Endpoint (Welcome & Status Dashboard)
# ============================================================
@app.get("/")
async def root():
    """
    Returns the server status and basic configuration properties.
    """
    return {
        "status": "Online",
        "message": "ESP32-P4 Voice Assistant Core Backend is active.",
        "configuration": {
            "server_ip": config.SERVER_IP,
            "port": config.SERVER_PORT,
            "whisper_model": config.WHISPER_MODEL,
            "whisper_device": config.WHISPER_DEVICE,
            "llm_model": config.DEEPSEEK_MODEL,
            "tts_voice": config.KOKORO_VOICE,
            "target_pcm_rate": f"{config.AUDIO_SAMPLE_RATE}Hz Mono 16-bit"
        }
    }


# ============================================================
# 2. Standalone Speech-to-Text Endpoint (/transcribe)
# ============================================================
@app.post("/transcribe", response_model=TranscribeResponse, responses={500: {"model": ErrorResponse}})
async def transcribe(file: UploadFile = File(...)):
    """
    Accepts an audio file upload from the ESP32 (or browser) in memory,
    runs inference through Faster-Whisper, and returns text results instantly.
    """
    logger.info(f"Incoming request on /transcribe from client: {file.filename}")
    try:
        # Read the file data straight into a memory buffer (Zero SSD space used)
        audio_bytes = await file.read()
        
        # Pass bytes to Whisper service (detects WAV wrapper automatically)
        stt_result = whisper_service.transcribe_audio(audio_bytes, is_raw_pcm=False)
        return stt_result
        
    except Exception as e:
        logger.error(f"Error executing standalone transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


# ============================================================
# 3. Standard Assistant Turn Endpoint (/assistant)
# ============================================================
@app.post("/assistant", response_model=AssistantResponse, responses={500: {"model": ErrorResponse}})
async def assistant_turn(file: UploadFile = File(...)):
    """
    Processes a standard full turn in one request.
    Takes incoming mic data, returns JSON text for both what it heard and the AI reply.
    """
    logger.info("Incoming request on full text endpoint /assistant")
    try:
        audio_bytes = await file.read()
        pipeline_result = await assistant_pipeline.execute_full_pipeline(
            audio_bytes=audio_bytes,
            is_raw_pcm=False,
            history=None # Expand here if using conversational state files later
        )
        return pipeline_result
    except Exception as e:
        logger.error(f"Error executing complete pipeline loop: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline loop failed: {str(e)}")


# ============================================================
# 4. Ultra-Low Latency Streaming Endpoint (/assistant_stream)
# ============================================================
@app.post("/assistant_stream")
async def assistant_voice_stream(file: UploadFile = File(...)):
    """
    The main performance endpoint for your assistant.
    1. Transcribes incoming audio and triggers DeepSeek.
    2. As soon as Kokoro generates the first complete sentence, it streams 
       raw 16kHz binary PCM blocks straight down the network socket back to the ESP32.
    """
    logger.info("Incoming request on streaming voice endpoint /assistant_stream")
    try:
        audio_bytes = await file.read()
        
        # Create a generator expression that executes our streaming pipeline
        audio_generator = assistant_pipeline.stream_assistant_audio(
            audio_bytes=audio_bytes,
            is_raw_pcm=False,
            history=None
        )
        
        # Return a continuous chunked binary transfer stream
        return StreamingResponse(
            audio_generator,
            media_type="application/octet-stream" # Identifies raw binary samples (no header)
        )
        
    except Exception as e:
        logger.error(f"Error handling live streaming pipeline iteration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Streaming loop failed: {str(e)}")


# ============================================================
# Execution Entry Point
# ============================================================
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Launching voice server engine on http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    
    # Run onport 8000 on all local subnet binds so your ESP32 can route to it freely
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)
