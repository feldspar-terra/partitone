"""Light cleanup processing for audio stems."""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Union
import pyloudnorm as pyln
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from app.audio_utils import (
    normalize_audio_shape,
    to_soundfile_shape,
    to_mono,
    load_audio_stems,
    save_audio,
)
from app.config import DEFAULT_TARGET_LUFS, DEFAULT_SILENCE_THRESHOLD_DB

logger = logging.getLogger(__name__)


class LightCleanup:
    """Light cleanup processing to preserve human element while handling technical issues."""
    
    def __init__(self, target_lufs: float = DEFAULT_TARGET_LUFS):
        """Initialize light cleanup processor.
        
        Args:
            target_lufs: Target loudness in LUFS for balancing (default: -23.0)
        """
        self.target_lufs = target_lufs
    
    def trim_silence(self, audio: np.ndarray, sr: int, threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB) -> np.ndarray:
        """Remove leading and trailing silence.
        
        Args:
            audio: Audio signal (mono or stereo)
            sr: Sample rate
            threshold_db: Silence threshold in dB (default: -60.0)
        
        Returns:
            Trimmed audio signal in original format
        """
        import librosa
        
        # Store original shape to preserve format
        original_shape = audio.shape
        is_mono_original = len(original_shape) == 1
        
        # Normalize to librosa format for consistent handling
        audio_normalized = normalize_audio_shape(audio)
        
        # Convert to mono for analysis
        audio_mono = to_mono(audio_normalized)
        
        # Trim silence using librosa
        trimmed_mono, (start_idx, end_idx) = librosa.effects.trim(
            audio_mono,
            top_db=abs(threshold_db),
            frame_length=2048,
            hop_length=512
        )
        
        # Apply same trim indices to original audio
        if len(audio_normalized.shape) > 1:
            # Trim stereo channels using same indices
            trimmed = audio_normalized[:, start_idx:end_idx]
        else:
            trimmed = trimmed_mono.reshape(1, -1)  # Ensure 2D for consistency
        
        # Return in original format
        if is_mono_original:
            return trimmed[0] if len(trimmed.shape) > 1 else trimmed
        else:
            # Check if original was (n_samples, n_channels) or (n_channels, n_samples)
            if original_shape[0] < original_shape[1]:
                # Original was (n_samples, n_channels), transpose back
                return trimmed.T
            else:
                # Original was (n_channels, n_samples), return as-is
                return trimmed
    
    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset from audio signal.
        
        Args:
            audio: Audio signal
        
        Returns:
            Audio signal with DC offset removed (in-place operation when possible)
        """
        audio = audio.copy()  # Make a copy to avoid modifying input
        normalized = normalize_audio_shape(audio)
        
        if len(normalized.shape) > 1:
            # Process each channel separately (in-place)
            for ch in range(normalized.shape[0]):
                normalized[ch] -= np.mean(normalized[ch])
        else:
            normalized[0] -= np.mean(normalized[0])
        
        # Return in original shape
        return normalize_audio_shape(normalized) if audio.shape != normalized.shape else normalized
    
    def phase_align(self, stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
        """Phase align stems to reduce phase cancellation using FFT-based cross-correlation.
        
        Args:
            stems: Dictionary of stem names to audio arrays
            sr: Sample rate
        
        Returns:
            Phase-aligned stems dictionary
        """
        if len(stems) < 2:
            return stems
        
        # Convert all to mono for phase analysis
        mono_stems = {}
        for name, audio in stems.items():
            mono_stems[name] = to_mono(audio)
        
        # Use vocals as reference (if available)
        reference_name = 'vocals' if 'vocals' in mono_stems else list(mono_stems.keys())[0]
        reference = mono_stems[reference_name]
        
        aligned_stems = {}
        for name, audio in stems.items():
            if name == reference_name:
                aligned_stems[name] = audio
                continue
            
            mono = mono_stems[name]
            
            # FFT-based cross-correlation for better performance
            # Pad to avoid circular correlation
            n = len(reference) + len(mono) - 1
            n_fft = 2 ** int(np.ceil(np.log2(n)))
            
            ref_fft = np.fft.fft(reference, n_fft)
            mono_fft = np.fft.fft(mono, n_fft)
            correlation = np.fft.ifft(ref_fft * np.conj(mono_fft)).real
            
            max_corr_idx = np.argmax(correlation)
            offset = max_corr_idx - (len(reference) - 1)
            
            # Early exit if already aligned
            if abs(offset) < 10:  # Less than 10 samples offset
                aligned_stems[name] = audio
                continue
            
            # Apply phase correction (simple time shift)
            if abs(offset) < len(mono) // 10:  # Only apply if offset is reasonable
                audio_normalized = normalize_audio_shape(audio)
                
                if offset > 0:
                    # Shift forward: pad at start
                    aligned = np.pad(audio_normalized, ((0, 0), (offset, 0)), mode='constant')[:, :audio_normalized.shape[1]]
                elif offset < 0:
                    # Shift backward: pad at end
                    aligned = np.pad(audio_normalized, ((0, 0), (0, -offset)), mode='constant')[:, -audio_normalized.shape[1]:]
                else:
                    aligned = audio_normalized
                
                aligned_stems[name] = aligned
            else:
                aligned_stems[name] = audio
        
        return aligned_stems
    
    def loudness_balance(self, stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
        """Soft stem-level loudness balancing.
        
        Args:
            stems: Dictionary of stem names to audio arrays
            sr: Sample rate
        
        Returns:
            Loudness-balanced stems dictionary
        """
        meter = pyln.Meter(sr)
        balanced_stems = {}
        
        for name, audio in stems.items():
            # Convert to mono for loudness measurement
            audio_mono = to_mono(audio)
            
            # Measure loudness
            try:
                loudness = meter.integrated_loudness(audio_mono)
                
                # Calculate gain adjustment (soft limiting)
                if not np.isnan(loudness) and loudness != -np.inf:
                    gain_db = self.target_lufs - loudness
                    # Soft limit: don't adjust more than 6dB
                    gain_db = np.clip(gain_db, -6.0, 6.0)
                    gain_linear = 10 ** (gain_db / 20.0)
                    
                    # Apply gain (in-place multiplication)
                    balanced = audio * gain_linear
                    
                    # Prevent clipping
                    max_val = np.max(np.abs(balanced))
                    if max_val > 0.95:
                        balanced *= (0.95 / max_val)
                    
                    balanced_stems[name] = balanced
                else:
                    balanced_stems[name] = audio
            except Exception as e:
                logger.warning(f"Failed to balance loudness for {name}: {e}")
                balanced_stems[name] = audio
        
        return balanced_stems
    
    def process_stems(self, stems_dict: Union[Dict[str, str], Dict[str, np.ndarray]], 
                     sr: int = 44100, format: str = "wav") -> Union[Dict[str, str], Dict[str, np.ndarray]]:
        """Process stems with light cleanup.
        
        Args:
            stems_dict: Dictionary mapping stem names to file paths or audio arrays
            sr: Sample rate (will be detected from files if paths provided)
            format: Output format ("wav" or "flac") - only used if saving to files
        
        Returns:
            Dictionary mapping stem names to processed file paths or audio arrays
            (same type as input)
        """
        logger.info("Applying light cleanup to stems...")
        
        # Check if input is file paths or audio arrays
        is_file_paths = isinstance(next(iter(stems_dict.values())), (str, Path))
        
        if is_file_paths:
            # Load all stems from files
            loaded_stems, detected_sr = load_audio_stems(stems_dict, sr=None)
            sr = detected_sr or sr
            
            if not loaded_stems:
                logger.warning("No stems loaded for cleanup")
                return stems_dict
        else:
            # Input is already audio arrays
            loaded_stems = stems_dict
            # Use provided sample rate, don't override it
            if sr is None:
                logger.warning("Sample rate not provided for audio arrays, using 44100")
                sr = 44100
        
        # Apply cleanup stages with parallel processing for independent operations
        num_stems = len(loaded_stems)
        max_workers = min(num_stems, multiprocessing.cpu_count())
        
        # Parallelize trim_silence and remove_dc_offset (independent per stem)
        def process_stem_cleanup(name_audio_pair):
            name, audio = name_audio_pair
            trimmed = self.trim_silence(audio, sr)
            cleaned = self.remove_dc_offset(trimmed)
            return name, cleaned
        
        logger.info("Trimming silence and removing DC offset...")
        if num_stems > 1 and max_workers > 1:
            # Process stems in parallel
            logger.info(f"Processing {num_stems} stems in parallel (max_workers={max_workers})...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                cleaned_stems_list = list(executor.map(process_stem_cleanup, loaded_stems.items()))
            cleaned_stems = dict(cleaned_stems_list)
            logger.info("Parallel stem processing complete")
        else:
            # Sequential processing for single stem or single CPU
            logger.debug(f"Sequential processing: {num_stems} stem(s), max_workers={max_workers}")
            cleaned_stems = {name: self.remove_dc_offset(self.trim_silence(audio, sr)) 
                           for name, audio in loaded_stems.items()}
        
        # Loudness balance needs all stems together (not parallelizable)
        logger.info("Balancing loudness...")
        cleaned_stems = self.loudness_balance(cleaned_stems, sr)
        
        if is_file_paths:
            # Save processed stems to files
            processed_paths = {}
            for name, audio in cleaned_stems.items():
                original_path = Path(stems_dict[name])
                output_path = original_path  # Overwrite original
                
                try:
                    save_audio(audio, output_path, sr, format)
                    processed_paths[name] = str(output_path)
                    logger.info(f"Saved cleaned {name} -> {output_path}")
                except Exception as e:
                    logger.error(f"Failed to save cleaned {name}: {e}")
                    processed_paths[name] = stems_dict[name]  # Fallback to original
            
            logger.info("Light cleanup complete")
            return processed_paths
        else:
            # Return audio arrays
            logger.info("Light cleanup complete")
            return cleaned_stems

