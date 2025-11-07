"""Unit tests for custom exceptions."""

import pytest

from app.exceptions import (
    PartiToneError,
    AudioProcessingError,
    StemNotFoundError,
    InvalidAudioFormatError,
    DemucsExecutionError,
    FileSizeLimitError,
)


class TestExceptions:
    """Tests for custom exception classes."""
    
    def test_partitone_error_base(self):
        """Test PartiToneError base exception."""
        error = PartiToneError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_audio_processing_error(self):
        """Test AudioProcessingError."""
        error = AudioProcessingError("Processing failed")
        assert str(error) == "Processing failed"
        assert isinstance(error, PartiToneError)
        assert isinstance(error, Exception)
    
    def test_stem_not_found_error(self):
        """Test StemNotFoundError."""
        error = StemNotFoundError("Stem not found: vocals")
        assert str(error) == "Stem not found: vocals"
        assert isinstance(error, PartiToneError)
    
    def test_invalid_audio_format_error(self):
        """Test InvalidAudioFormatError."""
        error = InvalidAudioFormatError("Unsupported format: mp4")
        assert str(error) == "Unsupported format: mp4"
        assert isinstance(error, PartiToneError)
    
    def test_demucs_execution_error(self):
        """Test DemucsExecutionError."""
        error = DemucsExecutionError("Demucs failed")
        assert str(error) == "Demucs failed"
        assert isinstance(error, PartiToneError)
    
    def test_file_size_limit_error(self):
        """Test FileSizeLimitError."""
        error = FileSizeLimitError("File too large: 600MB")
        assert str(error) == "File too large: 600MB"
        assert isinstance(error, PartiToneError)
    
    def test_exception_inheritance(self):
        """Test exception inheritance hierarchy."""
        # All custom exceptions should inherit from PartiToneError
        assert issubclass(AudioProcessingError, PartiToneError)
        assert issubclass(StemNotFoundError, PartiToneError)
        assert issubclass(InvalidAudioFormatError, PartiToneError)
        assert issubclass(DemucsExecutionError, PartiToneError)
        assert issubclass(FileSizeLimitError, PartiToneError)
        
        # All should be Exception subclasses
        assert issubclass(PartiToneError, Exception)
    
    def test_exception_raising(self):
        """Test that exceptions can be raised and caught."""
        with pytest.raises(AudioProcessingError):
            raise AudioProcessingError("Test")
        
        with pytest.raises(PartiToneError):
            raise StemNotFoundError("Test")

