"""Glitch repair processing for audio stems."""

import logging
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, Union, Optional
from scipy import signal, interpolate
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from app.audio_utils import (
    normalize_audio_shape,
    to_mono,
    load_audio_stems,
    save_audio,
)
from app.config import (
    DEFAULT_CLICK_THRESHOLD, DEFAULT_STUTTER_THRESHOLD, DEFAULT_SAMPLE_RATE,
    GLITCH_REPAIR_FAST_CLICK_THRESHOLD, GLITCH_REPAIR_FAST_STUTTER_THRESHOLD,
    GLITCH_REPAIR_STANDARD_CLICK_THRESHOLD, GLITCH_REPAIR_STANDARD_STUTTER_THRESHOLD,
    GLITCH_REPAIR_THOROUGH_CLICK_THRESHOLD, GLITCH_REPAIR_THOROUGH_STUTTER_THRESHOLD,
)

logger = logging.getLogger(__name__)


class GlitchRepair:
    """Glitch repair to reduce minor digital artifacts."""
    
    def __init__(self, click_threshold: Optional[float] = None, 
                 stutter_threshold: Optional[float] = None, 
                 repair_mode: str = "standard",
                 sr: int = DEFAULT_SAMPLE_RATE):
        """Initialize glitch repair processor.
        
        Args:
            click_threshold: Threshold for click detection (overrides repair_mode if provided)
            stutter_threshold: Threshold for stutter detection (overrides repair_mode if provided)
            repair_mode: Repair mode ("fast", "standard", "thorough", or "none")
            sr: Sample rate for filter caching (default: 44100)
        """
        # Set thresholds based on repair mode if not explicitly provided
        if click_threshold is None:
            if repair_mode == "fast":
                click_threshold = GLITCH_REPAIR_FAST_CLICK_THRESHOLD
            elif repair_mode == "thorough":
                click_threshold = GLITCH_REPAIR_THOROUGH_CLICK_THRESHOLD
            else:  # standard or default
                click_threshold = GLITCH_REPAIR_STANDARD_CLICK_THRESHOLD
        
        if stutter_threshold is None:
            if repair_mode == "fast":
                stutter_threshold = GLITCH_REPAIR_FAST_STUTTER_THRESHOLD
            elif repair_mode == "thorough":
                stutter_threshold = GLITCH_REPAIR_THOROUGH_STUTTER_THRESHOLD
            else:  # standard or default
                stutter_threshold = GLITCH_REPAIR_STANDARD_STUTTER_THRESHOLD
        
        self.click_threshold = click_threshold
        self.stutter_threshold = stutter_threshold
        self.repair_mode = repair_mode
        self.sr = sr
        self._click_filter_sos: Optional[np.ndarray] = None
        self._cached_sr: Optional[int] = None
        
        logger.debug(f"GlitchRepair initialized with mode={repair_mode}, "
                    f"click_threshold={click_threshold}, stutter_threshold={stutter_threshold}")
    
    def _get_click_filter(self, sr: int) -> np.ndarray:
        """Get or create cached high-pass filter for click detection.
        
        Args:
            sr: Sample rate
            
        Returns:
            Filter SOS coefficients
        """
        if self._click_filter_sos is None or self._cached_sr != sr:
            self._click_filter_sos = signal.butter(4, 2000, btype='high', fs=sr, output='sos')
            self._cached_sr = sr
        return self._click_filter_sos
    
    def detect_clicks(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Detect clicks and pops in audio using vectorized operations.
        
        Args:
            audio: Audio signal
            sr: Sample rate
        
        Returns:
            Boolean array indicating click locations
        """
        # Convert to mono for analysis
        audio_mono = to_mono(audio)
        
        # High-pass filter to emphasize transients (use cached filter)
        sos = self._get_click_filter(sr)
        filtered = signal.sosfilt(sos, audio_mono)
        
        # Vectorized detection of sudden amplitude changes (clicks)
        diff = np.abs(np.diff(filtered, prepend=filtered[0]))
        threshold = np.percentile(diff, 99.5) * self.click_threshold
        
        # Vectorized mask creation
        click_mask = diff > threshold
        
        return click_mask
    
    def detect_stutters(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Detect stutters and micro-glitches.
        
        Args:
            audio: Audio signal
            sr: Sample rate
        
        Returns:
            Boolean array indicating stutter locations
        """
        # Convert to mono for analysis
        audio_mono = to_mono(audio)
        
        # Short-time energy analysis
        frame_length = int(0.01 * sr)  # 10ms frames
        hop_length = frame_length // 2
        
        energy = librosa.feature.rms(
            y=audio_mono,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Vectorized detection of sudden energy drops (stutters)
        energy_diff = np.abs(np.diff(energy, prepend=energy[0]))
        threshold = np.percentile(energy_diff, 95) * self.stutter_threshold
        
        stutter_frames = np.where(energy_diff > threshold)[0]
        
        # Vectorized conversion of frame indices to sample indices
        stutter_mask = np.zeros(len(audio_mono), dtype=bool)
        for frame_idx in stutter_frames:
            start = frame_idx * hop_length
            end = min(start + frame_length, len(audio_mono))
            stutter_mask[start:end] = True
        
        return stutter_mask
    
    def repair_clicks(self, audio: np.ndarray, sr: int, click_mask: np.ndarray) -> np.ndarray:
        """Repair clicks and pops using interpolation.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            click_mask: Boolean array indicating click locations
        
        Returns:
            Repaired audio signal
        """
        repaired = audio.copy()
        audio_normalized = normalize_audio_shape(audio)
        
        # Process each channel
        window_ms = 0.005
        window_samples = int(window_ms * sr)
        
        for ch in range(audio_normalized.shape[0]):
            channel = audio_normalized[ch].copy()
            click_locations = np.where(click_mask)[0]
            
            if len(click_locations) == 0:
                continue
            
            # Group nearby clicks to avoid redundant interpolation
            click_groups = []
            if len(click_locations) > 0:
                current_start = click_locations[0]
                current_end = click_locations[0]
                
                for click_idx in click_locations[1:]:
                    if click_idx - current_end <= window_samples * 2:
                        current_end = click_idx
                    else:
                        click_groups.append((max(0, current_start - window_samples), 
                                           min(len(channel), current_end + window_samples)))
                        current_start = click_idx
                        current_end = click_idx
                click_groups.append((max(0, current_start - window_samples), 
                                   min(len(channel), current_end + window_samples)))
            
            # Interpolate over click regions
            x = np.arange(len(channel))
            for start, end in click_groups:
                if end - start > 2:
                    mask = np.ones(len(channel), dtype=bool)
                    mask[start:end] = False
                    
                    if np.sum(mask) > 1:
                        interp_func = interpolate.interp1d(
                            x[mask],
                            channel[mask],
                            kind='linear',
                            fill_value='extrapolate'
                        )
                        channel[start:end] = interp_func(x[start:end])
            
            audio_normalized[ch] = channel
        
        # Return in original shape
        return audio_normalized
    
    def repair_stutters(self, audio: np.ndarray, sr: int, stutter_mask: np.ndarray) -> np.ndarray:
        """Repair stutters using crossfade.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            stutter_mask: Boolean array indicating stutter locations
        
        Returns:
            Repaired audio signal
        """
        repaired = normalize_audio_shape(audio.copy())
        
        # Find contiguous stutter regions (vectorized)
        diff_mask = np.diff(stutter_mask.astype(int), prepend=0, append=0)
        starts = np.where(diff_mask == 1)[0]
        ends = np.where(diff_mask == -1)[0]
        stutter_regions = list(zip(starts, ends))
        
        # Process each channel
        for ch in range(repaired.shape[0]):
            channel = repaired[ch]
            
            # Repair each stutter region
            for start_idx, end_idx in stutter_regions:
                if end_idx - start_idx < 10:  # Skip very short regions
                    continue
                
                # Crossfade repair: blend with surrounding audio
                fade_length = min(int(0.01 * sr), (end_idx - start_idx) // 2)  # 10ms or half region
                
                if start_idx > fade_length and end_idx < len(channel) - fade_length:
                    # Create smooth transition
                    fade_in = np.linspace(0, 1, fade_length)
                    fade_out = np.linspace(1, 0, fade_length)
                    
                    # Blend with surrounding audio
                    before_region = channel[start_idx - fade_length:start_idx]
                    after_region = channel[end_idx:end_idx + fade_length]
                    
                    # Interpolate middle section
                    middle_length = end_idx - start_idx
                    interpolated = np.linspace(
                        channel[start_idx - 1],
                        channel[end_idx],
                        middle_length + 2
                    )[1:-1]
                    
                    # Apply crossfades (in-place operations)
                    channel[start_idx - fade_length:start_idx] = (
                        before_region * (1 - fade_in) + 
                        np.full(fade_length, channel[start_idx - fade_length]) * fade_in
                    )
                    channel[start_idx:end_idx] = interpolated
                    channel[end_idx:end_idx + fade_length] = (
                        after_region * fade_out +
                        np.full(fade_length, channel[end_idx + fade_length]) * (1 - fade_out)
                    )
            
            repaired[ch] = channel
        
        return repaired
    
    def process_stems(self, stems_dict: Union[Dict[str, str], Dict[str, np.ndarray]], 
                     sr: int = 44100, format: str = "wav") -> Union[Dict[str, str], Dict[str, np.ndarray]]:
        """Process stems with glitch repair.
        
        Args:
            stems_dict: Dictionary mapping stem names to file paths or audio arrays
            sr: Sample rate (will be detected from files if paths provided)
            format: Output format ("wav" or "flac") - only used if saving to files
        
        Returns:
            Dictionary mapping stem names to processed file paths or audio arrays
            (same type as input)
        """
        logger.info("Applying glitch repair to stems...")
        
        # Check if input is file paths or audio arrays
        is_file_paths = isinstance(next(iter(stems_dict.values())), (str, Path))
        
        if is_file_paths:
            # Load all stems from files
            loaded_stems, detected_sr = load_audio_stems(stems_dict, sr=None)
            sr = detected_sr or sr
            
            if not loaded_stems:
                logger.warning("No stems loaded for glitch repair")
                return stems_dict
        else:
            # Input is already audio arrays
            loaded_stems = stems_dict
            if sr is None or sr == 44100:
                logger.warning("Sample rate not provided for audio arrays, using 44100")
                sr = 44100
        
        # Skip repair if mode is "none"
        if self.repair_mode == "none":
            logger.info("Glitch repair mode is 'none', skipping repair")
            return stems_dict
        
        # Repair each stem (can be parallelized since stems are independent)
        num_stems = len(loaded_stems)
        max_workers = min(num_stems, multiprocessing.cpu_count())
        
        def repair_single_stem(name_audio_pair):
            """Repair a single stem (helper for parallel processing)."""
            name, audio = name_audio_pair
            logger.info(f"Repairing glitches in {name} (mode: {self.repair_mode})...")
            
            # Detect clicks
            click_mask = self.detect_clicks(audio, sr)
            click_count = np.sum(click_mask)
            if click_count > 0:
                logger.info(f"  Detected {click_count} potential clicks in {name}")
                audio = self.repair_clicks(audio, sr, click_mask)
            else:
                logger.debug(f"  No clicks detected in {name}")
            
            # Detect stutters
            stutter_mask = self.detect_stutters(audio, sr)
            stutter_count = np.sum(stutter_mask)
            if stutter_count > 0:
                logger.info(f"  Detected {stutter_count} potential stutters in {name}")
                audio = self.repair_stutters(audio, sr, stutter_mask)
            else:
                logger.debug(f"  No stutters detected in {name}")
            
            return name, audio
        
        if num_stems > 1 and max_workers > 1:
            # Process stems in parallel
            logger.info(f"Processing {num_stems} stems in parallel (max_workers={max_workers})...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                repaired_stems_list = list(executor.map(repair_single_stem, loaded_stems.items()))
            repaired_stems = dict(repaired_stems_list)
        else:
            # Sequential processing for single stem or single CPU
            repaired_stems = {}
            for name, audio in loaded_stems.items():
                repaired_name, repaired_audio = repair_single_stem((name, audio))
                repaired_stems[repaired_name] = repaired_audio
        
        if is_file_paths:
            # Save repaired stems to files
            processed_paths = {}
            for name, audio in repaired_stems.items():
                original_path = Path(stems_dict[name])
                output_path = original_path  # Overwrite original
                
                try:
                    save_audio(audio, output_path, sr, format)
                    processed_paths[name] = str(output_path)
                    logger.info(f"Saved repaired {name} -> {output_path}")
                except (IOError, OSError, ValueError) as e:
                    logger.error(f"Failed to save repaired {name}: {e}")
                    processed_paths[name] = stems_dict[name]  # Fallback to original
            
            logger.info("Glitch repair complete")
            return processed_paths
        else:
            # Return audio arrays
            logger.info("Glitch repair complete")
            return repaired_stems

