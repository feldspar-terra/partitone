"""Configuration defaults for PartiTone."""

# Light Cleanup defaults
DEFAULT_TARGET_LUFS = -23.0
DEFAULT_SILENCE_THRESHOLD_DB = -60.0

# Glitch Repair defaults
DEFAULT_CLICK_THRESHOLD = 0.1
DEFAULT_STUTTER_THRESHOLD = 0.15

# Audio processing defaults
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_OUTPUT_FORMAT = "wav"

# Demucs defaults
DEFAULT_MODEL = "htdemucs"

# API defaults
MAX_FILE_SIZE_MB = 500
API_TIMEOUT_SECONDS = 3600  # 1 hour

# Batch processing defaults
BATCH_PROGRESS_UPDATE_INTERVAL = 1  # Update progress every N files

 