"""Unit tests for audio utility functions."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import soundfile as sf

from app.audio_utils import (
    is_stereo,
    normalize_audio_shape,
    to_soundfile_shape,
    to_mono,
    load_audio_stems,
    save_audio,
)


class TestIsStereo:
    """Tests for is_stereo function."""
    
    def test_mono_1d(self):
        """Test that 1D array is not stereo."""
        audio = np.array([1.0, 2.0, 3.0])
        assert is_stereo(audio) is False
    
    def test_mono_2d_single_channel(self):
        """Test that 2D array with 1 channel is not stereo."""
        audio = np.array([[1.0, 2.0, 3.0]])  # (1, n_samples)
        assert is_stereo(audio) is False
    
    def test_stereo_channels_first(self):
        """Test that stereo with channels first is detected."""
        audio = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, n_samples)
        assert is_stereo(audio) is True
    
    def test_stereo_samples_first(self):
        """Test that stereo with samples first is detected."""
        audio = np.array([[1.0, 3.0], [2.0, 4.0]])  # (n_samples, 2)
        assert is_stereo(audio) is True
    
    def test_multi_channel(self):
        """Test that multi-channel (>2) is detected as stereo."""
        audio = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, n_samples)
        assert is_stereo(audio) is True


class TestNormalizeAudioShape:
    """Tests for normalize_audio_shape function."""
    
    def test_mono_1d(self):
        """Test that 1D mono is converted to (1, n_samples)."""
        audio = np.array([1.0, 2.0, 3.0])
        normalized = normalize_audio_shape(audio)
        assert normalized.shape == (1, 3)
    
    def test_stereo_channels_first(self):
        """Test that stereo channels-first stays the same."""
        audio = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, n_samples)
        normalized = normalize_audio_shape(audio)
        assert normalized.shape == (2, 2)
        np.testing.assert_array_equal(normalized, audio)
    
    def test_stereo_samples_first(self):
        """Test that stereo samples-first is transposed."""
        audio = np.array([[1.0, 3.0], [2.0, 4.0]])  # (n_samples, 2)
        normalized = normalize_audio_shape(audio)
        assert normalized.shape == (2, 2)
        np.testing.assert_array_equal(normalized, np.array([[1.0, 2.0], [3.0, 4.0]]))


class TestToSoundfileShape:
    """Tests for to_soundfile_shape function."""
    
    def test_mono_1d(self):
        """Test that 1D mono is converted to (n_samples, 1)."""
        audio = np.array([1.0, 2.0, 3.0])
        sf_shape = to_soundfile_shape(audio)
        assert sf_shape.shape == (3, 1)
    
    def test_stereo_channels_first(self):
        """Test that stereo channels-first is transposed."""
        audio = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, n_samples)
        sf_shape = to_soundfile_shape(audio)
        assert sf_shape.shape == (2, 2)
        np.testing.assert_array_equal(sf_shape, np.array([[1.0, 3.0], [2.0, 4.0]]))
    
    def test_stereo_samples_first(self):
        """Test that stereo samples-first stays the same."""
        audio = np.array([[1.0, 3.0], [2.0, 4.0]])  # (n_samples, 2)
        sf_shape = to_soundfile_shape(audio)
        assert sf_shape.shape == (2, 2)
        np.testing.assert_array_equal(sf_shape, audio)


class TestToMono:
    """Tests for to_mono function."""
    
    def test_mono_1d(self):
        """Test that 1D mono stays mono."""
        audio = np.array([1.0, 2.0, 3.0])
        mono = to_mono(audio)
        assert mono.shape == (3,)
        np.testing.assert_array_equal(mono, audio)
    
    def test_stereo_averaging(self):
        """Test that stereo is averaged to mono."""
        audio = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, n_samples)
        mono = to_mono(audio)
        assert mono.shape == (2,)
        np.testing.assert_array_equal(mono, np.array([2.0, 3.0]))  # Average of channels
    
    def test_stereo_samples_first(self):
        """Test that stereo samples-first is converted correctly."""
        audio = np.array([[1.0, 3.0], [2.0, 4.0]])  # (n_samples, 2)
        mono = to_mono(audio)
        assert mono.shape == (2,)
        np.testing.assert_array_almost_equal(mono, np.array([2.0, 3.5]), decimal=5)


class TestLoadAudioStems:
    """Tests for load_audio_stems function."""
    
    def test_load_valid_files(self, temp_dir):
        """Test loading valid audio files."""
        # Create test audio files
        stem1_path = temp_dir / "vocals.wav"
        stem2_path = temp_dir / "drums.wav"
        
        # Create minimal valid WAV files
        sample_rate = 44100
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create simple sine waves
        audio1 = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        audio2 = np.sin(2 * np.pi * 880 * t)  # 880 Hz tone
        
        sf.write(str(stem1_path), audio1, sample_rate)
        sf.write(str(stem2_path), audio2, sample_rate)
        
        stems_dict = {
            "vocals": str(stem1_path),
            "drums": str(stem2_path),
        }
        
        loaded_stems, detected_sr = load_audio_stems(stems_dict, sr=None)
        
        assert "vocals" in loaded_stems
        assert "drums" in loaded_stems
        assert detected_sr == sample_rate
        assert loaded_stems["vocals"].shape[0] == 1  # Mono
        assert loaded_stems["drums"].shape[0] == 1
    
    def test_load_with_sample_rate(self, temp_dir):
        """Test loading with specified sample rate."""
        stem_path = temp_dir / "test.wav"
        sample_rate = 22050
        audio = np.random.randn(sample_rate)  # 1 second of noise
        sf.write(str(stem_path), audio, sample_rate)
        
        stems_dict = {"test": str(stem_path)}
        loaded_stems, detected_sr = load_audio_stems(stems_dict, sr=44100)
        
        assert detected_sr == 44100  # Resampled
        assert loaded_stems["test"].shape[1] == 44100  # 1 second at 44.1kHz
    
    def test_load_nonexistent_file(self):
        """Test that nonexistent file is handled gracefully."""
        stems_dict = {"nonexistent": "/path/to/nonexistent.wav"}
        loaded_stems, detected_sr = load_audio_stems(stems_dict, sr=None)
        
        assert "nonexistent" not in loaded_stems
        assert detected_sr == 44100  # Default


class TestSaveAudio:
    """Tests for save_audio function."""
    
    def test_save_mono(self, temp_dir):
        """Test saving mono audio."""
        output_path = temp_dir / "output.wav"
        sample_rate = 44100
        audio = np.random.randn(sample_rate)  # 1 second of noise
        
        save_audio(audio, output_path, sample_rate, format="wav")
        
        assert output_path.exists()
        loaded, sr = sf.read(str(output_path))
        assert sr == sample_rate
        assert len(loaded.shape) == 1  # Mono
    
    def test_save_stereo(self, temp_dir):
        """Test saving stereo audio."""
        output_path = temp_dir / "output_stereo.wav"
        sample_rate = 44100
        audio = np.random.randn(2, sample_rate)  # Stereo
        
        save_audio(audio, output_path, sample_rate, format="wav")
        
        assert output_path.exists()
        loaded, sr = sf.read(str(output_path))
        assert sr == sample_rate
        assert loaded.shape[1] == 2  # Stereo
    
    def test_save_flac(self, temp_dir):
        """Test saving FLAC format."""
        output_path = temp_dir / "output.flac"
        sample_rate = 44100
        audio = np.random.randn(sample_rate)
        
        save_audio(audio, output_path, sample_rate, format="flac")
        
        assert output_path.exists()
        loaded, sr = sf.read(str(output_path))
        assert sr == sample_rate
    
    def test_save_different_shapes(self, temp_dir):
        """Test that different input shapes are handled."""
        sample_rate = 44100
        audio_1d = np.random.randn(sample_rate)
        audio_2d_channels = np.random.randn(2, sample_rate)
        audio_2d_samples = np.random.randn(sample_rate, 2)
        
        # All should save successfully
        save_audio(audio_1d, temp_dir / "1d.wav", sample_rate)
        save_audio(audio_2d_channels, temp_dir / "2d_channels.wav", sample_rate)
        save_audio(audio_2d_samples, temp_dir / "2d_samples.wav", sample_rate)
        
        assert (temp_dir / "1d.wav").exists()
        assert (temp_dir / "2d_channels.wav").exists()
        assert (temp_dir / "2d_samples.wav").exists()

