"""Utility functions for PartiTone."""

import os
import re
from pathlib import Path
from typing import List, Optional


# Supported audio file extensions
SUPPORTED_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.aac'}

# Stem name mapping
STEM_NAMES = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']


def is_audio_file(file_path: str) -> bool:
    """Check if a file is a supported audio format."""
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


def get_audio_files(directory: str) -> List[str]:
    """Get all audio files from a directory."""
    directory_path = Path(directory)
    if not directory_path.is_dir():
        return []
    
    audio_files = []
    for file_path in directory_path.iterdir():
        if file_path.is_file() and is_audio_file(str(file_path)):
            audio_files.append(str(file_path))
    
    return sorted(audio_files)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for use in paths."""
    # Remove extension
    name = Path(filename).stem
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[^\w\s-]', '', name)
    # Replace spaces and multiple underscores with single underscore
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    return sanitized


def create_output_directory(base_output: str, track_name: str) -> str:
    """Create output directory for a track."""
    sanitized_name = sanitize_filename(track_name)
    output_dir = Path(base_output) / sanitized_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def get_stem_path(output_dir: str, stem_name: str, format: str = 'wav') -> str:
    """Get the full path for a stem file."""
    stem_filename = f"{stem_name}.{format}"
    return str(Path(output_dir) / stem_filename)


def validate_input_path(input_path: str) -> bool:
    """Validate that input path exists."""
    path = Path(input_path)
    return path.exists()


def format_audio_path(input_path: str) -> str:
    """Normalize and format audio file path."""
    return str(Path(input_path).resolve())


def get_output_format(format_str: str) -> str:
    """Validate and return output format."""
    format_str = format_str.lower()
    if format_str not in ['wav', 'flac']:
        return 'wav'
    return format_str


def ensure_directory(path: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)

