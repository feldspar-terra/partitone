"""Configuration defaults for PartiTone."""

import os
from typing import Dict, Literal, Optional
from dataclasses import dataclass

# Light Cleanup defaults
DEFAULT_TARGET_LUFS = -23.0
DEFAULT_SILENCE_THRESHOLD_DB = -60.0

# Glitch Repair defaults
DEFAULT_CLICK_THRESHOLD = 0.1
DEFAULT_STUTTER_THRESHOLD = 0.15

# Glitch Repair mode thresholds
GLITCH_REPAIR_FAST_CLICK_THRESHOLD = 0.15  # Higher threshold = fewer detections = faster
GLITCH_REPAIR_FAST_STUTTER_THRESHOLD = 0.25
GLITCH_REPAIR_STANDARD_CLICK_THRESHOLD = 0.1
GLITCH_REPAIR_STANDARD_STUTTER_THRESHOLD = 0.15
GLITCH_REPAIR_THOROUGH_CLICK_THRESHOLD = 0.05  # Lower threshold = more detections = thorough
GLITCH_REPAIR_THOROUGH_STUTTER_THRESHOLD = 0.1

# Audio processing defaults
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_OUTPUT_FORMAT = "wav"

# Demucs defaults
DEFAULT_MODEL = "htdemucs"

# API defaults - can be overridden via environment variables
MAX_FILE_SIZE_MB = int(os.getenv("PARTITONE_MAX_FILE_SIZE_MB", "500"))
API_TIMEOUT_SECONDS = int(os.getenv("PARTITONE_API_TIMEOUT_SECONDS", "3600"))  # 1 hour
SSE_POLL_INTERVAL = float(os.getenv("PARTITONE_SSE_POLL_INTERVAL", "0.5"))  # SSE polling interval in seconds

# Batch processing defaults
BATCH_PROGRESS_UPDATE_INTERVAL = int(os.getenv("PARTITONE_BATCH_PROGRESS_INTERVAL", "1"))  # Update progress every N files
MAX_PARALLEL_WORKERS: Optional[int] = None  # None = auto-detect (CPU count), or set to specific number
if os.getenv("PARTITONE_MAX_PARALLEL_WORKERS"):
    MAX_PARALLEL_WORKERS = int(os.getenv("PARTITONE_MAX_PARALLEL_WORKERS"))

# Job cleanup defaults
JOB_CLEANUP_INTERVAL_HOURS = int(os.getenv("PARTITONE_JOB_CLEANUP_INTERVAL_HOURS", "1"))  # Clean up jobs older than N hours
JOB_CLEANUP_ENABLED = os.getenv("PARTITONE_JOB_CLEANUP_ENABLED", "true").lower() == "true"

# Performance presets
PresetName = Literal["fast", "balanced", "quality", "ultra"]


@dataclass
class PerformancePreset:
    """Performance preset configuration."""
    name: str
    model: str
    raw: bool
    repair_glitch: bool
    repair_mode: str  # "fast", "standard", "thorough", or "none"
    description: str
    estimated_time_3min_song: str  # Estimated processing time for 3-minute song


PERFORMANCE_PRESETS: Dict[PresetName, PerformancePreset] = {
    "fast": PerformancePreset(
        name="fast",
        model="htdemucs",
        raw=True,
        repair_glitch=False,
        repair_mode="none",
        description="Fastest processing: Raw Demucs output, no cleanup or repair. Best for quick previews.",
        estimated_time_3min_song="5-15 seconds (GPU)"
    ),
    "balanced": PerformancePreset(
        name="balanced",
        model="htdemucs",
        raw=False,
        repair_glitch=False,
        repair_mode="none",
        description="Balanced quality and speed: Light cleanup applied, no glitch repair. Recommended default.",
        estimated_time_3min_song="15-30 seconds (GPU)"
    ),
    "quality": PerformancePreset(
        name="quality",
        model="htdemucs_ft",
        raw=False,
        repair_glitch=True,
        repair_mode="standard",
        description="Higher quality: Fine-tuned model with cleanup and standard glitch repair.",
        estimated_time_3min_song="30-60 seconds (GPU)"
    ),
    "ultra": PerformancePreset(
        name="ultra",
        model="htdemucs_ft",
        raw=False,
        repair_glitch=True,
        repair_mode="thorough",
        description="Ultra quality: Fine-tuned model with cleanup and thorough glitch repair. Best quality, slower processing.",
        estimated_time_3min_song="60-90 seconds (GPU)"
    ),
}

DEFAULT_PRESET: PresetName = "balanced"


def get_preset(preset_name: str) -> PerformancePreset:
    """Get performance preset by name.
    
    Args:
        preset_name: Name of preset (fast, balanced, quality, ultra)
        
    Returns:
        PerformancePreset configuration
        
    Raises:
        ValueError: If preset name is invalid
    """
    preset_name_lower = preset_name.lower()
    if preset_name_lower not in PERFORMANCE_PRESETS:
        valid_presets = ", ".join(PERFORMANCE_PRESETS.keys())
        raise ValueError(f"Invalid preset '{preset_name}'. Valid presets: {valid_presets}")
    return PERFORMANCE_PRESETS[preset_name_lower]

 