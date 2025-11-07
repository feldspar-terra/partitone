"""Base class for audio processors."""

from abc import ABC, abstractmethod
from typing import Dict, Union
from pathlib import Path
import numpy as np

from app.audio_utils import load_audio_stems, save_audio
from app.config import DEFAULT_SAMPLE_RATE


class BaseAudioProcessor(ABC):
    """Abstract base class for audio stem processors."""
    
    def __init__(self, sr: int = DEFAULT_SAMPLE_RATE):
        """Initialize processor.
        
        Args:
            sr: Sample rate for processing
        """
        self.sr = sr
    
    @abstractmethod
    def process_audio(self, stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
        """Process audio stems (to be implemented by subclasses).
        
        Args:
            stems: Dictionary mapping stem names to audio arrays
            sr: Sample rate
        
        Returns:
            Dictionary mapping stem names to processed audio arrays
        """
        pass
    
    def load_stems(self, stems_dict: Dict[str, Union[str, Path]], sr: int = None) -> tuple:
        """Load stems from file paths.
        
        Args:
            stems_dict: Dictionary mapping stem names to file paths
            sr: Target sample rate (None to use file's native rate)
        
        Returns:
            Tuple of (loaded_stems_dict, detected_sample_rate)
        """
        return load_audio_stems(stems_dict, sr=sr)
    
    def save_stems(self, stems: Dict[str, np.ndarray], output_paths: Dict[str, Union[str, Path]], 
                   sr: int, format: str = "wav") -> Dict[str, str]:
        """Save processed stems to files.
        
        Args:
            stems: Dictionary mapping stem names to audio arrays
            output_paths: Dictionary mapping stem names to output file paths
            sr: Sample rate
            format: Output format ("wav" or "flac")
        
        Returns:
            Dictionary mapping stem names to saved file paths
        """
        saved_paths = {}
        for name, audio in stems.items():
            if name in output_paths:
                save_audio(audio, output_paths[name], sr, format)
                saved_paths[name] = str(output_paths[name])
        return saved_paths
    
    def process_stems(self, stems_dict: Union[Dict[str, str], Dict[str, np.ndarray]], 
                     sr: int = None, format: str = "wav") -> Union[Dict[str, str], Dict[str, np.ndarray]]:
        """Process stems (handles both file paths and audio arrays).
        
        Args:
            stems_dict: Dictionary mapping stem names to file paths or audio arrays
            sr: Sample rate (will be detected from files if paths provided)
            format: Output format ("wav" or "flac") - only used if saving to files
        
        Returns:
            Dictionary mapping stem names to processed file paths or audio arrays
            (same type as input)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Check if input is file paths or audio arrays
        is_file_paths = isinstance(next(iter(stems_dict.values())), (str, Path))
        
        if is_file_paths:
            # Load all stems from files
            loaded_stems, detected_sr = self.load_stems(stems_dict, sr=None)
            sr = detected_sr or sr or self.sr
            
            if not loaded_stems:
                logger.warning("No stems loaded for processing")
                return stems_dict
        else:
            # Input is already audio arrays
            loaded_stems = stems_dict
            sr = sr or self.sr
        
        # Process stems (implemented by subclass)
        processed_stems = self.process_audio(loaded_stems, sr)
        
        if is_file_paths:
            # Save processed stems to files
            processed_paths = {}
            for name, audio in processed_stems.items():
                if name in stems_dict:
                    original_path = Path(stems_dict[name])
                    output_path = original_path  # Overwrite original
                    
                    try:
                        save_audio(audio, output_path, sr, format)
                        processed_paths[name] = str(output_path)
                    except Exception as e:
                        logger.error(f"Failed to save processed {name}: {e}")
                        processed_paths[name] = stems_dict[name]  # Fallback to original
            
            return processed_paths
        else:
            # Return audio arrays
            return processed_stems

