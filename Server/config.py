"""
============================================================
ESP32-P4 AI Voice Assistant
AI Server Configuration

Backend:
    FastAPI
    Faster-Whisper
    DeepSeek
    Kokoro TTS

============================================================
"""

import os


# ============================================================
# Server Configuration
# ============================================================

SERVER_HOST = "10.157.42.233"

SERVER_PORT = 8000


# ============================================================
# ESP32 Connection
# ============================================================

# ESP32 will connect to:
#
# http://10.157.42.233:8000

SERVER_IP = "10.157.42.233"



# ============================================================
# Audio Configuration
# ============================================================

AUDIO_SAMPLE_RATE = 16000

AUDIO_CHANNELS = 1

AUDIO_BITS = 16



# ============================================================
# Whisper Configuration
# ============================================================

# Options:
#
# tiny
# base
# small
# medium
# large-v3
#
# For ESP32 assistant:
#
# small  -> good balance
# medium -> better accuracy

WHISPER_MODEL = "small"


WHISPER_DEVICE = "cpu"


WHISPER_COMPUTE_TYPE = "int8"



# ============================================================
# DeepSeek Configuration
# ============================================================

# Add your API key here

DEEPSEEK_API_KEY = os.getenv(
    "DEEPSEEK_API_KEY",
    "YOUR_DEEPSEEK_API_KEY"
)


DEEPSEEK_MODEL = "deepseek-chat"



# ============================================================
# Kokoro TTS Configuration
# ============================================================

KOKORO_SAMPLE_RATE = 16000


KOKORO_VOICE = "af_heart"



# ============================================================
# Streaming Configuration
# ============================================================

# Audio chunk size sent to ESP32

STREAM_CHUNK_SIZE = 4096



# ============================================================
# Temporary Storage
# ============================================================

# Debug only
#
# Normal operation:
# Audio stays in RAM

TEMP_FOLDER = "temp"
