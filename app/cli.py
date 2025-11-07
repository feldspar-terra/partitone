"""CLI interface for PartiTone."""

import argparse
import sys
import logging
from pathlib import Path

from app.runner import DemucsRunner
from app.utils import (
    validate_input_path,
    is_audio_file,
    get_audio_files,
    ensure_directory,
)
from app.config import get_preset, DEFAULT_PRESET


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PartiTone: Audio stem separation tool using Demucs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Input audio file or directory (for batch processing)',
    )
    
    parser.add_argument(
        '--stems',
        type=int,
        default=6,
        help='Number of stems to extract (default: 6)',
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='htdemucs',
        help='Model to use (default: htdemucs)',
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='/output',
        help='Output directory (default: /output)',
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['wav', 'flac'],
        default='wav',
        help='Output format (default: wav)',
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process entire directory (input must be a directory)',
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use (default: auto)',
    )
    
    parser.add_argument(
        '--raw',
        action='store_true',
        help='Skip light cleanup entirely (output raw Demucs stems)',
    )
    
    parser.add_argument(
        '--repair-glitch',
        action='store_true',
        help='Enable glitch repair stage (reduce clicks, pops, stutters)',
    )
    
    parser.add_argument(
        '--preset',
        type=str,
        choices=['fast', 'balanced', 'quality', 'ultra'],
        default=None,
        help=f'Performance preset (fast/balanced/quality/ultra). Overrides --model, --raw, --repair-glitch. Default: {DEFAULT_PRESET}',
    )
    
    parser.add_argument(
        '--repair-mode',
        type=str,
        choices=['fast', 'standard', 'thorough'],
        default='standard',
        help='Glitch repair mode when --repair-glitch is enabled (default: standard)',
    )
    
    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Validate input path
    if not validate_input_path(args.input):
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Apply preset if provided (preset overrides individual settings)
    if args.preset:
        try:
            preset_config = get_preset(args.preset)
            model = preset_config.model
            raw = preset_config.raw
            repair_glitch = preset_config.repair_glitch
            repair_mode = preset_config.repair_mode
            logger.info(f"Using preset '{args.preset}': {preset_config.description}")
            logger.info(f"  Estimated time for 3min song: {preset_config.estimated_time_3min_song}")
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)
    else:
        # Use command-line arguments or defaults
        model = args.model
        raw = args.raw
        repair_glitch = args.repair_glitch
        repair_mode = args.repair_mode
    
    # Determine device
    device = None if args.device == 'auto' else args.device
    
    # Initialize runner
    try:
        runner = DemucsRunner(model=model, device=device)
    except Exception as e:
        logger.error(f"Failed to initialize Demucs runner: {e}")
        sys.exit(1)
    
    # Ensure output directory exists
    ensure_directory(args.output)
    
    # Determine if batch or single file
    input_path = Path(args.input)
    is_batch_mode = args.batch or input_path.is_dir()
    
    if is_batch_mode:
        if not input_path.is_dir():
            logger.error("Batch mode requires a directory as input")
            sys.exit(1)
        
        logger.info(f"Batch processing mode: {args.input}")
        audio_files = get_audio_files(args.input)
        
        if not audio_files:
            logger.error(f"No audio files found in {args.input}")
            sys.exit(1)
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        results = runner.separate_batch(
            args.input, 
            args.output, 
            args.format,
            raw=raw,
            repair_glitch=repair_glitch,
            repair_mode=repair_mode
        )
        
        successful = sum(1 for r in results if 'error' not in r)
        failed = len(results) - successful
        
        logger.info(f"Batch processing complete: {successful} successful, {failed} failed")
        
        if failed > 0:
            sys.exit(1)
    else:
        if not input_path.is_file():
            logger.error(f"Input is not a file: {args.input}")
            sys.exit(1)
        
        if not is_audio_file(args.input):
            logger.error(f"Unsupported audio format: {args.input}")
            sys.exit(1)
        
        logger.info(f"Processing single file: {args.input}")
        try:
            result = runner.separate(
                args.input, 
                args.output, 
                args.format,
                raw=raw,
                repair_glitch=repair_glitch,
                repair_mode=repair_mode
            )
            logger.info(f"Successfully processed: {result['output_dir']}")
            
            # Display performance metrics if available
            if 'performance_metrics' in result:
                metrics = result['performance_metrics']
                logger.info("Performance metrics:")
                logger.info(f"  Total time: {metrics.get('total_time', 0):.2f}s")
                logger.info(f"  Separation: {metrics.get('separation_time', 0):.2f}s")
                if 'cleanup_time' in metrics:
                    logger.info(f"  Cleanup: {metrics.get('cleanup_time', 0):.2f}s")
                if 'repair_time' in metrics:
                    logger.info(f"  Repair: {metrics.get('repair_time', 0):.2f}s")
        except Exception as e:
            logger.error(f"Failed to process {args.input}: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()

