"""Pytest configuration and fixtures for PartiTone tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator
from fastapi.testclient import TestClient

from app.api import app
from app.background import job_manager


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_client() -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_audio_file(temp_dir: Path) -> Path:
    """Create a mock audio file for testing."""
    # Create a simple WAV file header (minimal valid WAV file)
    audio_path = temp_dir / "test_audio.wav"
    
    # Create a minimal valid WAV file (44 bytes header + minimal data)
    wav_header = (
        b'RIFF'  # ChunkID
        b'\x24\x00\x00\x00'  # ChunkSize (36 bytes)
        b'WAVE'  # Format
        b'fmt '  # Subchunk1ID
        b'\x10\x00\x00\x00'  # Subchunk1Size (16 bytes)
        b'\x01\x00'  # AudioFormat (PCM)
        b'\x01\x00'  # NumChannels (mono)
        b'\x44\xac\x00\x00'  # SampleRate (44100)
        b'\x88\x58\x01\x00'  # ByteRate
        b'\x02\x00'  # BlockAlign
        b'\x10\x00'  # BitsPerSample (16)
        b'data'  # Subchunk2ID
        b'\x00\x00\x00\x00'  # Subchunk2Size (0 bytes - empty)
    )
    
    with open(audio_path, 'wb') as f:
        f.write(wav_header)
    
    return audio_path


@pytest.fixture(autouse=True)
def reset_job_manager():
    """Reset job manager before each test."""
    # Clear all jobs before test
    job_manager.jobs.clear()
    yield
    # Clean up after test
    job_manager.jobs.clear()

