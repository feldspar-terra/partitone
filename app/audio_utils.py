"""Audio utility functions for shape handling and loading."""

import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, Tuple, Union

logger = logging.getLogger(__name__)


def is_stereo(audio: np.ndarray) -> bool:
    """Check if audio array is stereo.
    
    Args:
        audio: Audio array (any shape)
        
    Returns:
        True if audio is stereo (has 2 dimensions and 2 channels)
    """
    return len(audio.shape) > 1 and (audio.shape[0] == 2 or audio.shape[1] == 2)


def normalize_audio_shape(audio: np.ndarray) -> np.ndarray:
    """Convert audio to librosa format: (n_channels, n_samples).
    
    Args:
        audio: Audio array in any format
        
    Returns:
        Audio array in librosa format (n_channels, n_samples)
    """
    if len(audio.shape) == 1:
        # Mono: reshape to (1, n_samples)
        return audio.reshape(1, -1)
    elif audio.shape[0] < audio.shape[1]:
        # Shape is (n_samples, n_channels), transpose to (n_channels, n_samples)
        return audio.T
    else:
        # Already in (n_channels, n_samples) format
        return audio


def to_soundfile_shape(audio: np.ndarray) -> np.ndarray:
    """Convert audio to soundfile format: (n_samples, n_channels).
    
    Args:
        audio: Audio array in any format
        
    Returns:
        Audio array in soundfile format (n_samples, n_channels)
    """
    if len(audio.shape) == 1:
        # Mono: reshape to (n_samples, 1)
        return audio.reshape(-1, 1)
    elif audio.shape[0] <= 2 and audio.shape[0] < audio.shape[1]:
        # Shape is (n_channels, n_samples), transpose to (n_samples, n_channels)
        return audio.T
    else:
        # Already in (n_samples, n_channels) format
        return audio


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo audio to mono for analysis.
    
    Args:
        audio: Audio array in any format
        
    Returns:
        Mono audio array (1D)
    """
    normalized = normalize_audio_shape(audio)
    
    if normalized.shape[0] == 1:
        # Already mono
        return normalized[0]
    else:
        # Stereo: average channels
        return np.mean(normalized, axis=0)


def load_audio_stems(stems_dict: Dict[str, Union[str, Path]], sr: int = None) -> Tuple[Dict[str, np.ndarray], int]:
    """Load audio stems from file paths.
    
    Args:
        stems_dict: Dictionary mapping stem names to file paths
        sr: Target sample rate (None to use file's native rate)
        
    Returns:
        Tuple of (loaded_stems_dict, detected_sample_rate)
    """
    loaded_stems = {}
    detected_sr = None
    
    for name, path in stems_dict.items():
        try:
            audio, sample_rate = librosa.load(str(path), sr=sr, mono=False)
            loaded_stems[name] = audio
            if detected_sr is None:
                detected_sr = sample_rate
            elif detected_sr != sample_rate:
                logger.warning(f"Sample rate mismatch: {name} has {sample_rate}, expected {detected_sr}")
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            continue
    
    return loaded_stems, detected_sr or (sr if sr else 44100)


def save_audio(audio: np.ndarray, output_path: Union[str, Path], sr: int, format: str = "wav") -> None:
    """Save audio array to file.
    
    Args:
        audio: Audio array in any format
        output_path: Path to save file
        sr: Sample rate
        format: Output format ("wav" or "flac")
    """
    # Convert to soundfile format
    audio_sf = to_soundfile_shape(audio)
    
    # Save using soundfile
    sf.write(str(output_path), audio_sf, sr, format=format.upper())

