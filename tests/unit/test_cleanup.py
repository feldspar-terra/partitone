"""Unit tests for LightCleanup class."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import soundfile as sf

from app.cleanup import LightCleanup
from app.config import DEFAULT_TARGET_LUFS, DEFAULT_SILENCE_THRESHOLD_DB


class TestLightCleanup:
    """Tests for LightCleanup class."""
    
    def test_initialization(self):
        """Test LightCleanup initialization."""
        cleanup = LightCleanup()
        assert cleanup.target_lufs == DEFAULT_TARGET_LUFS
        
        cleanup_custom = LightCleanup(target_lufs=-20.0)
        assert cleanup_custom.target_lufs == -20.0
    
    def test_trim_silence_mono(self):
        """Test trimming silence from mono audio."""
        cleanup = LightCleanup()
        sr = 44100
        
        # Create audio with silence at start and end
        silence_samples = int(0.1 * sr)  # 100ms silence
        audio_samples = int(0.5 * sr)  # 500ms audio
        
        audio = np.concatenate([
            np.zeros(silence_samples),  # Leading silence
            np.random.randn(audio_samples) * 0.5,  # Audio content
            np.zeros(silence_samples)  # Trailing silence
        ])
        
        trimmed = cleanup.trim_silence(audio, sr)
        
        # Trimmed audio should be shorter
        assert len(trimmed) < len(audio)
        # Should have removed significant silence
        assert len(trimmed) <= audio_samples + silence_samples
    
    def test_trim_silence_stereo(self):
        """Test trimming silence from stereo audio."""
        cleanup = LightCleanup()
        sr = 44100
        
        silence_samples = int(0.1 * sr)
        audio_samples = int(0.5 * sr)
        
        # Create stereo audio
        audio = np.array([
            np.concatenate([np.zeros(silence_samples), np.random.randn(audio_samples) * 0.5, np.zeros(silence_samples)]),
            np.concatenate([np.zeros(silence_samples), np.random.randn(audio_samples) * 0.5, np.zeros(silence_samples)])
        ])
        
        trimmed = cleanup.trim_silence(audio, sr)
        
        # Should preserve stereo shape
        assert len(trimmed.shape) == 2
        assert trimmed.shape[0] == 2  # Stereo channels
    
    def test_remove_dc_offset(self):
        """Test DC offset removal."""
        cleanup = LightCleanup()
        
        # Create audio with DC offset
        audio = np.random.randn(1000) + 0.5  # DC offset of 0.5
        cleaned = cleanup.remove_dc_offset(audio)
        
        # Mean should be close to zero
        assert np.abs(np.mean(cleaned)) < 0.01
    
    def test_remove_dc_offset_stereo(self):
        """Test DC offset removal from stereo audio."""
        cleanup = LightCleanup()
        
        # Create stereo audio with different DC offsets per channel
        audio = np.array([
            np.random.randn(1000) + 0.5,  # Channel 1 with offset
            np.random.randn(1000) - 0.3   # Channel 2 with offset
        ])
        
        cleaned = cleanup.remove_dc_offset(audio)
        
        # Both channels should have mean close to zero
        assert np.abs(np.mean(cleaned[0])) < 0.01
        assert np.abs(np.mean(cleaned[1])) < 0.01
    
    def test_phase_align(self):
        """Test phase alignment of stems."""
        cleanup = LightCleanup()
        sr = 44100
        
        # Create two stems with slight phase offset
        t = np.linspace(0, 0.1, int(sr * 0.1))
        stem1 = np.sin(2 * np.pi * 440 * t)
        stem2 = np.sin(2 * np.pi * 440 * t + np.pi / 4)  # Phase offset
        
        stems = {
            "vocals": stem1,
            "drums": stem2
        }
        
        aligned = cleanup.phase_align(stems, sr)
        
        # Should return aligned stems
        assert "vocals" in aligned
        assert "drums" in aligned
        assert len(aligned) == 2
    
    def test_phase_align_single_stem(self):
        """Test phase alignment with single stem."""
        cleanup = LightCleanup()
        sr = 44100
        
        stems = {
            "vocals": np.random.randn(1000)
        }
        
        aligned = cleanup.phase_align(stems, sr)
        
        # Should return unchanged
        assert aligned == stems
    
    def test_loudness_balance(self):
        """Test loudness balancing."""
        cleanup = LightCleanup()
        sr = 44100
        
        # Create stems with different loudness
        quiet_stem = np.random.randn(int(sr * 0.1)) * 0.1  # Quiet
        loud_stem = np.random.randn(int(sr * 0.1)) * 0.8   # Loud
        
        stems = {
            "vocals": quiet_stem,
            "drums": loud_stem
        }
        
        balanced = cleanup.loudness_balance(stems, sr)
        
        # Should return balanced stems
        assert "vocals" in balanced
        assert "drums" in balanced
        # Loudness should be more similar (soft limiting applied)
        assert np.max(np.abs(balanced["vocals"])) <= 1.0
        assert np.max(np.abs(balanced["drums"])) <= 1.0
    
    def test_process_stems_arrays(self):
        """Test processing stems as audio arrays."""
        cleanup = LightCleanup()
        sr = 44100
        
        stems = {
            "vocals": np.random.randn(int(sr * 0.1)),
            "drums": np.random.randn(int(sr * 0.1))
        }
        
        processed = cleanup.process_stems(stems, sr=sr)
        
        # Should return processed stems
        assert "vocals" in processed
        assert "drums" in processed
        assert isinstance(processed["vocals"], np.ndarray)
        assert isinstance(processed["drums"], np.ndarray)
    
    def test_process_stems_files(self, temp_dir):
        """Test processing stems from file paths."""
        cleanup = LightCleanup()
        sr = 44100
        
        # Create test audio files
        vocals_path = temp_dir / "vocals.wav"
        drums_path = temp_dir / "drums.wav"
        
        audio = np.random.randn(int(sr * 0.1))
        sf.write(str(vocals_path), audio, sr)
        sf.write(str(drums_path), audio, sr)
        
        stems = {
            "vocals": str(vocals_path),
            "drums": str(drums_path)
        }
        
        processed = cleanup.process_stems(stems, sr=sr)
        
        # Should return file paths
        assert "vocals" in processed
        assert "drums" in processed
        assert isinstance(processed["vocals"], str)
        assert Path(processed["vocals"]).exists()
    
    def test_process_stems_empty(self):
        """Test processing empty stems dictionary."""
        cleanup = LightCleanup()
        
        processed = cleanup.process_stems({}, sr=44100)
        
        assert processed == {}
    
    def test_trim_silence_no_silence(self):
        """Test trimming when there's no silence."""
        cleanup = LightCleanup()
        sr = 44100
        
        # Create audio without silence
        audio = np.random.randn(int(sr * 0.1)) * 0.5
        
        trimmed = cleanup.trim_silence(audio, sr)
        
        # Should still return valid audio
        assert len(trimmed) > 0
    
    def test_remove_dc_offset_no_offset(self):
        """Test DC offset removal when there's no offset."""
        cleanup = LightCleanup()
        
        # Create audio without DC offset
        audio = np.random.randn(1000)
        cleaned = cleanup.remove_dc_offset(audio)
        
        # Should still be valid
        assert len(cleaned) == len(audio)

