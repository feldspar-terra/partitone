"""Unit tests for utility functions."""

import pytest
from pathlib import Path
from app.utils import (
    is_audio_file,
    get_audio_files,
    sanitize_filename,
    create_output_directory,
    get_stem_path,
    validate_input_path,
    format_audio_path,
    get_output_format,
    ensure_directory,
)


class TestIsAudioFile:
    """Tests for is_audio_file function."""
    
    def test_valid_extensions(self):
        """Test that valid audio extensions are recognized."""
        assert is_audio_file("song.mp3") is True
        assert is_audio_file("song.wav") is True
        assert is_audio_file("song.flac") is True
        assert is_audio_file("song.m4a") is True
        assert is_audio_file("song.aac") is True
    
    def test_case_insensitive(self):
        """Test that extension matching is case insensitive."""
        assert is_audio_file("song.MP3") is True
        assert is_audio_file("song.WAV") is True
        assert is_audio_file("song.FLAC") is True
    
    def test_invalid_extensions(self):
        """Test that invalid extensions are rejected."""
        assert is_audio_file("song.txt") is False
        assert is_audio_file("song.pdf") is False
        assert is_audio_file("song") is False
        assert is_audio_file("song.") is False
    
    def test_path_with_directories(self):
        """Test that paths with directories work correctly."""
        assert is_audio_file("/path/to/song.mp3") is True
        assert is_audio_file("path/to/song.wav") is True


class TestGetAudioFiles:
    """Tests for get_audio_files function."""
    
    def test_empty_directory(self, temp_dir):
        """Test that empty directory returns empty list."""
        assert get_audio_files(str(temp_dir)) == []
    
    def test_directory_with_audio_files(self, temp_dir):
        """Test that audio files are found."""
        # Create test audio files
        (temp_dir / "song1.mp3").touch()
        (temp_dir / "song2.wav").touch()
        (temp_dir / "song3.flac").touch()
        (temp_dir / "not_audio.txt").touch()
        
        files = get_audio_files(str(temp_dir))
        assert len(files) == 3
        assert any("song1.mp3" in f for f in files)
        assert any("song2.wav" in f for f in files)
        assert any("song3.flac" in f for f in files)
        assert not any("not_audio.txt" in f for f in files)
    
    def test_nonexistent_directory(self):
        """Test that nonexistent directory returns empty list."""
        assert get_audio_files("/nonexistent/path") == []
    
    def test_file_path_instead_of_directory(self, temp_dir):
        """Test that file path returns empty list."""
        test_file = temp_dir / "test.mp3"
        test_file.touch()
        assert get_audio_files(str(test_file)) == []


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""
    
    def test_simple_filename(self):
        """Test sanitization of simple filename."""
        assert sanitize_filename("song.mp3") == "song"
    
    def test_special_characters(self):
        """Test that special characters are removed."""
        assert sanitize_filename("song (remix).mp3") == "song_remix"
        assert sanitize_filename("song@artist.mp3") == "songartist"
        assert sanitize_filename("song#1.mp3") == "song1"
    
    def test_spaces_and_underscores(self):
        """Test that spaces and underscores are normalized."""
        assert sanitize_filename("song name.mp3") == "song_name"
        assert sanitize_filename("song__name.mp3") == "song_name"
        assert sanitize_filename("song  name.mp3") == "song_name"
    
    def test_multiple_underscores(self):
        """Test that multiple underscores are collapsed."""
        assert sanitize_filename("song___name.mp3") == "song_name"


class TestCreateOutputDirectory:
    """Tests for create_output_directory function."""
    
    def test_directory_creation(self, temp_dir):
        """Test that output directory is created."""
        output_dir = create_output_directory(str(temp_dir), "song.mp3")
        assert Path(output_dir).exists()
        assert Path(output_dir).is_dir()
    
    def test_sanitized_name(self, temp_dir):
        """Test that filename is sanitized in directory name."""
        output_dir = create_output_directory(str(temp_dir), "song (remix).mp3")
        assert "song_remix" in output_dir
    
    def test_nested_directories(self, temp_dir):
        """Test that nested directories are created."""
        nested_base = temp_dir / "level1" / "level2"
        output_dir = create_output_directory(str(nested_base), "song.mp3")
        assert Path(output_dir).exists()


class TestGetStemPath:
    """Tests for get_stem_path function."""
    
    def test_path_generation(self, temp_dir):
        """Test that stem path is generated correctly."""
        output_dir = str(temp_dir)
        stem_path = get_stem_path(output_dir, "vocals", "wav")
        assert stem_path == str(temp_dir / "vocals.wav")
    
    def test_different_formats(self, temp_dir):
        """Test that different formats work."""
        output_dir = str(temp_dir)
        assert get_stem_path(output_dir, "vocals", "flac") == str(temp_dir / "vocals.flac")
        assert get_stem_path(output_dir, "drums", "wav") == str(temp_dir / "drums.wav")


class TestValidateInputPath:
    """Tests for validate_input_path function."""
    
    def test_existing_file(self, temp_dir):
        """Test that existing file is validated."""
        test_file = temp_dir / "test.mp3"
        test_file.touch()
        assert validate_input_path(str(test_file)) is True
    
    def test_existing_directory(self, temp_dir):
        """Test that existing directory is validated."""
        assert validate_input_path(str(temp_dir)) is True
    
    def test_nonexistent_path(self):
        """Test that nonexistent path is rejected."""
        assert validate_input_path("/nonexistent/path") is False


class TestFormatAudioPath:
    """Tests for format_audio_path function."""
    
    def test_path_resolution(self, temp_dir):
        """Test that path is resolved correctly."""
        test_file = temp_dir / "test.mp3"
        test_file.touch()
        formatted = format_audio_path(str(test_file))
        assert Path(formatted).is_absolute()
    
    def test_relative_path(self, temp_dir):
        """Test that relative path is converted to absolute."""
        test_file = temp_dir / "test.mp3"
        test_file.touch()
        # Change to temp_dir to test relative path
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            formatted = format_audio_path("test.mp3")
            assert Path(formatted).is_absolute()
        finally:
            os.chdir(old_cwd)


class TestGetOutputFormat:
    """Tests for get_output_format function."""
    
    def test_valid_formats(self):
        """Test that valid formats are returned."""
        assert get_output_format("wav") == "wav"
        assert get_output_format("flac") == "flac"
    
    def test_case_insensitive(self):
        """Test that format matching is case insensitive."""
        assert get_output_format("WAV") == "wav"
        assert get_output_format("FLAC") == "flac"
        assert get_output_format("Wav") == "wav"
    
    def test_invalid_format_defaults_to_wav(self):
        """Test that invalid format defaults to wav."""
        assert get_output_format("mp3") == "wav"
        assert get_output_format("invalid") == "wav"
        assert get_output_format("") == "wav"


class TestEnsureDirectory:
    """Tests for ensure_directory function."""
    
    def test_create_new_directory(self, temp_dir):
        """Test that new directory is created."""
        new_dir = temp_dir / "new_dir"
        ensure_directory(str(new_dir))
        assert new_dir.exists()
        assert new_dir.is_dir()
    
    def test_existing_directory(self, temp_dir):
        """Test that existing directory doesn't cause error."""
        ensure_directory(str(temp_dir))
        assert temp_dir.exists()
    
    def test_nested_directories(self, temp_dir):
        """Test that nested directories are created."""
        nested = temp_dir / "level1" / "level2" / "level3"
        ensure_directory(str(nested))
        assert nested.exists()
        assert nested.is_dir()

