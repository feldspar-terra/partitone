"""Core audio separation runner using Demucs."""

import os
import logging
import torch
import time
from pathlib import Path
from typing import List, Optional, Dict
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from app.utils import (
    create_output_directory,
    get_stem_path,
    format_audio_path,
    get_output_format,
    is_audio_file,
    get_audio_files,
)
from app.cleanup import LightCleanup
from app.glitch_repair import GlitchRepair
from app.audio_utils import load_audio_stems, save_audio
from app.config import MAX_PARALLEL_WORKERS
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemucsRunner:
    """Runner for Demucs audio separation."""
    
    def __init__(self, model: str = "htdemucs", device: Optional[str] = None):
        """Initialize Demucs runner.
        
        Args:
            model: Model name to use (default: "htdemucs")
            device: Device to use ("cuda", "cpu", or None for auto-detection)
        """
        self.model_name = model
        
        # Detect device with multi-GPU support
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                device_count = torch.cuda.device_count()
                if device_count > 1:
                    # Use the first GPU (can be enhanced to select fastest GPU)
                    logger.info(f"GPU (CUDA) detected: {device_count} devices available, using device 0")
                    logger.info(f"GPU device name: {torch.cuda.get_device_name(0)}")
                else:
                    logger.info(f"GPU (CUDA) detected and enabled: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                logger.info("Using CPU (GPU not available)")
        else:
            self.device = device
        
        logger.info(f"Demucs runner initialized with model: {model}, device: {self.device}")
    
    def separate(self, input_path: str, output_dir: str, format: str = "wav", 
                 raw: bool = False, repair_glitch: bool = False, 
                 repair_mode: str = "standard") -> dict:
        """Separate audio file into stems.
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save separated stems
            format: Output format ("wav" or "flac")
            raw: Skip light cleanup (default: False)
            repair_glitch: Enable glitch repair (default: False)
            repair_mode: Glitch repair mode ("fast", "standard", "thorough", or "none")
        
        Returns:
            Dictionary with stem paths, metadata, and performance metrics
        """
        start_time = time.time()
        performance_metrics: Dict[str, float] = {}
        
        input_path = format_audio_path(input_path)
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if not is_audio_file(input_path):
            raise ValueError(f"Unsupported audio format: {input_path}")
        
        # Create output directory
        track_name = Path(input_path).name
        track_output_dir = create_output_directory(output_dir, track_name)
        format = get_output_format(format)
        
        logger.info(f"Separating {input_path} -> {track_output_dir}")
        
        # GPU memory cleanup before processing (if using CUDA)
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared GPU cache before processing")
        
        # Run separation using Demucs command-line interface
        try:
            # Create temporary directory for Demucs output
            temp_output = Path(track_output_dir).parent / f"demucs_temp_{Path(track_name).stem}"
            temp_output.mkdir(parents=True, exist_ok=True)
            
            # Build Demucs command
            cmd = [
                "python", "-m", "demucs.separate",
                "-o", str(temp_output),
                "-n", self.model_name,
            ]
            
            # Add format flag
            if format == "flac":
                cmd.append("--flac")
            # else default is wav
            
            # Add device flag
            if self.device == "cuda":
                cmd.extend(["-d", "cuda"])
            else:
                cmd.extend(["-d", "cpu"])
            
            # Add input file
            cmd.append(str(input_path))
            
            logger.info(f"Running Demucs: {' '.join(cmd)}")
            
            # Execute Demucs with timing
            separation_start = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            separation_time = time.time() - separation_start
            performance_metrics['separation_time'] = separation_time
            logger.info(f"Demucs separation completed in {separation_time:.2f} seconds")
            
            logger.info(f"Demucs output: {result.stdout}")
            if result.stderr:
                logger.info(f"Demucs stderr: {result.stderr}")
            
            # Demucs creates output structure: temp_output/htdemucs/track_name/stem.wav
            # Find the actual output directory
            model_output_dir = temp_output / self.model_name / Path(track_name).stem
            
            if not model_output_dir.exists():
                # Try alternative structure
                model_output_dir = temp_output / Path(track_name).stem
                if not model_output_dir.exists():
                    raise FileNotFoundError(f"Demucs output not found in expected location: {model_output_dir}")
            
            # Map Demucs outputs to our stem names
            # Standard htdemucs outputs: drums.wav, bass.wav, other.wav, vocals.wav
            demucs_stems = ['drums', 'bass', 'other', 'vocals']
            stem_paths = {}
            
            # Copy and rename stems
            for stem_name in demucs_stems:
                demucs_file = model_output_dir / f"{stem_name}.{format}"
                if demucs_file.exists():
                    output_stem_path = get_stem_path(track_output_dir, stem_name, format)
                    shutil.copy2(demucs_file, output_stem_path)
                    stem_paths[stem_name] = output_stem_path
                    logger.info(f"Saved {stem_name} -> {output_stem_path}")
                else:
                    logger.warning(f"Stem file not found: {demucs_file}")
            
            # Create placeholder files for stems not in Demucs output
            # (guitar, piano) - copy from 'other' stem
            for stem_name in ['guitar', 'piano']:
                if 'other' in stem_paths:
                    stem_path = get_stem_path(track_output_dir, stem_name, format)
                    shutil.copy2(stem_paths['other'], stem_path)
                    stem_paths[stem_name] = stem_path
                    logger.info(f"Created placeholder {stem_name} from 'other' stem")
            
            # Clean up temporary directory immediately after copying stems
            try:
                shutil.rmtree(temp_output)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")
            
            # Load stems once for processing (if cleanup or repair needed)
            processed_stems = None
            detected_sr = None
            
            if not raw or repair_glitch:
                # Load stems into memory
                loaded_stems, detected_sr = load_audio_stems(stem_paths, sr=None)
                processed_stems = loaded_stems
            
            # Apply light cleanup (unless --raw flag is set)
            if not raw and processed_stems:
                try:
                    cleanup_start = time.time()
                    cleanup = LightCleanup()
                    # Use detected sample rate, ensure it's not None
                    cleanup_sr = detected_sr if detected_sr else 44100
                    processed_stems = cleanup.process_stems(processed_stems, sr=cleanup_sr, format=format)
                    cleanup_time = time.time() - cleanup_start
                    performance_metrics['cleanup_time'] = cleanup_time
                    logger.info(f"Light cleanup applied in {cleanup_time:.2f} seconds (sample rate: {cleanup_sr} Hz)")
                except Exception as e:
                    logger.warning(f"Cleanup failed, continuing with raw stems: {e}")
                    # Fallback: reload from files
                    if processed_stems:
                        processed_stems, detected_sr = load_audio_stems(stem_paths, sr=None)
            
            # Apply glitch repair (if --repair-glitch flag is set)
            if repair_glitch and processed_stems:
                try:
                    repair_start = time.time()
                    repair = GlitchRepair(repair_mode=repair_mode)
                    processed_stems = repair.process_stems(processed_stems, sr=detected_sr or 44100, format=format)
                    repair_time = time.time() - repair_start
                    performance_metrics['repair_time'] = repair_time
                    logger.info(f"Glitch repair ({repair_mode}) applied in {repair_time:.2f} seconds")
                except Exception as e:
                    logger.warning(f"Glitch repair failed: {e}")
            
            # Save processed stems if they were modified in memory
            if processed_stems and (not raw or repair_glitch):
                save_sr = detected_sr if detected_sr else 44100
                for name, audio in processed_stems.items():
                    if name in stem_paths:
                        try:
                            # Verify audio is not empty before saving
                            if audio.size == 0:
                                logger.warning(f"Audio array for {name} is empty, skipping save")
                                continue
                            save_audio(audio, stem_paths[name], save_sr, format)
                            logger.info(f"Saved processed {name} (sample rate: {save_sr} Hz)")
                        except Exception as e:
                            logger.error(f"Failed to save processed {name}: {e}")
                            import traceback
                            logger.debug(traceback.format_exc())
            
            # Calculate total time
            total_time = time.time() - start_time
            performance_metrics['total_time'] = total_time
            
            # GPU memory cleanup after processing (if using CUDA)
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared GPU cache after processing")
            
            # Log performance summary
            logger.info(f"Separation complete: {track_output_dir}")
            logger.info(f"Performance summary - Total: {total_time:.2f}s, "
                       f"Separation: {performance_metrics.get('separation_time', 0):.2f}s, "
                       f"Cleanup: {performance_metrics.get('cleanup_time', 0):.2f}s, "
                       f"Repair: {performance_metrics.get('repair_time', 0):.2f}s")
            
            return {
                'output_dir': track_output_dir,
                'stems': stem_paths,
                'device': self.device,
                'performance_metrics': performance_metrics,
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Demucs process failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise RuntimeError(f"Separation failed: {e.stderr}")
        except Exception as e:
            logger.error(f"Error during separation: {e}")
            raise
    
    def separate_batch(self, input_dir: str, output_dir: str, format: str = "wav",
                       raw: bool = False, repair_glitch: bool = False, 
                       repair_mode: str = "standard",
                       show_progress: bool = True,
                       max_workers: Optional[int] = None) -> List[dict]:
        """Process multiple audio files in a directory with parallel processing.
        
        Args:
            input_dir: Directory containing audio files
            output_dir: Directory to save separated stems
            format: Output format ("wav" or "flac")
            raw: Skip light cleanup (default: False)
            repair_glitch: Enable glitch repair (default: False)
            repair_mode: Glitch repair mode (default: "standard")
            show_progress: Show progress bar (default: True)
            max_workers: Maximum number of parallel workers (None = auto-detect)
        
        Returns:
            List of results for each processed file (in input order)
        """
        audio_files = get_audio_files(input_dir)
        
        if not audio_files:
            logger.warning(f"No audio files found in {input_dir}")
            return []
        
        # Determine number of workers
        if max_workers is None:
            max_workers = MAX_PARALLEL_WORKERS
        if max_workers is None:
            # Auto-detect: use CPU count, but limit to 4 for GPU systems to avoid memory issues
            cpu_count = multiprocessing.cpu_count()
            if self.device == "cuda":
                max_workers = min(4, cpu_count)  # Limit GPU parallelization
            else:
                max_workers = cpu_count  # Use all CPUs for CPU processing
        
        # For single file or GPU with limited memory, use sequential processing
        if len(audio_files) == 1 or (self.device == "cuda" and max_workers == 1):
            logger.info(f"Processing {len(audio_files)} file(s) sequentially")
            return self._process_batch_sequential(audio_files, output_dir, format, 
                                                  raw, repair_glitch, repair_mode, show_progress)
        
        logger.info(f"Processing {len(audio_files)} files in parallel (max_workers={max_workers})")
        
        # Try to import tqdm for progress bar
        try:
            from tqdm import tqdm
            use_tqdm = show_progress
        except ImportError:
            use_tqdm = False
            if show_progress:
                logger.info("tqdm not available, progress bar disabled")
        
        # Process files in parallel
        results_dict = {}  # Track results by file path to maintain order
        futures = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            for audio_file in audio_files:
                future = executor.submit(
                    self._process_single_file,
                    audio_file, output_dir, format, raw, repair_glitch, repair_mode
                )
                futures[future] = audio_file
            
            # Collect results as they complete
            iterator = tqdm(futures.items(), desc="Processing", total=len(audio_files), unit="file") if use_tqdm else futures.items()
            
            for future in as_completed(futures):
                audio_file = futures[future]
                if use_tqdm:
                    iterator.set_postfix(file=Path(audio_file).name[:30])
                else:
                    logger.info(f"Completed: {Path(audio_file).name}")
                
                try:
                    result = future.result()
                    results_dict[audio_file] = result
                except Exception as e:
                    logger.error(f"Failed to process {audio_file}: {e}")
                    results_dict[audio_file] = {
                        'input_file': audio_file,
                        'error': str(e),
                    }
        
        # Return results in original file order
        results = [results_dict[audio_file] for audio_file in audio_files]
        
        successful = sum(1 for r in results if 'error' not in r)
        failed = len(results) - successful
        logger.info(f"Batch processing complete: {successful} successful, {failed} failed")
        return results
    
    def _process_batch_sequential(self, audio_files: List[str], output_dir: str, 
                                   format: str, raw: bool, repair_glitch: bool, 
                                   repair_mode: str, show_progress: bool) -> List[dict]:
        """Process files sequentially (helper for single file or when parallelization disabled)."""
        results = []
        
        # Try to import tqdm for progress bar
        try:
            from tqdm import tqdm
            use_tqdm = show_progress
        except ImportError:
            use_tqdm = False
        
        iterator = tqdm(audio_files, desc="Processing", unit="file") if use_tqdm else audio_files
        
        for audio_file in iterator:
            if use_tqdm:
                iterator.set_postfix(file=Path(audio_file).name[:30])
            else:
                logger.info(f"Processing file {len(results) + 1}/{len(audio_files)}: {Path(audio_file).name}")
            
            result = self._process_single_file(audio_file, output_dir, format, 
                                              raw, repair_glitch, repair_mode)
            results.append(result)
        
        return results
    
    def _process_single_file(self, audio_file: str, output_dir: str, format: str,
                            raw: bool, repair_glitch: bool, repair_mode: str) -> dict:
        """Process a single file (helper for parallel processing)."""
        try:
            return self.separate(audio_file, output_dir, format, 
                               raw=raw, repair_glitch=repair_glitch, repair_mode=repair_mode)
        except Exception as e:
            logger.error(f"Failed to process {audio_file}: {e}")
            return {
                'input_file': audio_file,
                'error': str(e),
            }


def get_device_info() -> dict:
    """Get device information including GPU details."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    
    # Add additional GPU information if available
    if torch.cuda.is_available():
        try:
            info['gpu_memory_total_mb'] = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            info['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
            info['gpu_memory_cached_mb'] = torch.cuda.memory_reserved(0) / (1024**2) if torch.cuda.is_available() else 0
        except Exception as e:
            logger.debug(f"Could not retrieve detailed GPU info: {e}")
    
    return info

