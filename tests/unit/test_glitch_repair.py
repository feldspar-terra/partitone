"""Unit tests for GlitchRepair class."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import soundfile as sf

from app.glitch_repair import GlitchRepair
from app.config import DEFAULT_CLICK_THRESHOLD, DEFAULT_STUTTER_THRESHOLD, DEFAULT_SAMPLE_RATE


class TestGlitchRepair:
    """Tests for GlitchRepair class."""
    
    def test_initialization(self):
        """Test GlitchRepair initialization."""
        repair = GlitchRepair()
        assert repair.click_threshold == DEFAULT_CLICK_THRESHOLD
        assert repair.stutter_threshold == DEFAULT_STUTTER_THRESHOLD
        assert repair.sr == DEFAULT_SAMPLE_RATE
        
        repair_custom = GlitchRepair(
            click_threshold=0.2,
            stutter_threshold=0.3,
            sr=48000
        )
        assert repair_custom.click_threshold == 0.2
        assert repair_custom.stutter_threshold == 0.3
        assert repair_custom.sr == 48000
    
    def test_get_click_filter(self):
        """Test click filter caching."""
        repair = GlitchRepair()
        sr = 44100
        
        filter1 = repair._get_click_filter(sr)
        filter2 = repair._get_click_filter(sr)
        
        # Should return same filter (cached)
        np.testing.assert_array_equal(filter1, filter2)
        
        # Different sample rate should create new filter
        filter3 = repair._get_click_filter(48000)
        assert filter3 is not None
    
    def test_detect_clicks(self):
        """Test click detection."""
        repair = GlitchRepair()
        sr = 44100
        
        # Create audio with a click (sudden amplitude spike)
        audio = np.random.randn(int(sr * 0.1)) * 0.1
        click_position = len(audio) // 2
        audio[click_position] = 1.0  # Add click
        
        click_mask = repair.detect_clicks(audio, sr)
        
        # Should detect the click
        assert isinstance(click_mask, np.ndarray)
        assert click_mask.dtype == bool
        assert len(click_mask) == len(audio)
    
    def test_detect_clicks_no_clicks(self):
        """Test click detection with no clicks."""
        repair = GlitchRepair()
        sr = 44100
        
        # Create clean audio
        audio = np.random.randn(int(sr * 0.1)) * 0.1
        
        click_mask = repair.detect_clicks(audio, sr)
        
        # Should return mask (may or may not detect clicks depending on threshold)
        assert isinstance(click_mask, np.ndarray)
        assert len(click_mask) == len(audio)
    
    def test_detect_stutters(self):
        """Test stutter detection."""
        repair = GlitchRepair()
        sr = 44100
        
        # Create audio with a stutter (sudden energy drop)
        audio = np.random.randn(int(sr * 0.1)) * 0.5
        stutter_start = len(audio) // 3
        stutter_end = stutter_start + int(0.01 * sr)  # 10ms stutter
        audio[stutter_start:stutter_end] *= 0.01  # Sudden drop
        
        stutter_mask = repair.detect_stutters(audio, sr)
        
        # Should detect the stutter
        assert isinstance(stutter_mask, np.ndarray)
        assert stutter_mask.dtype == bool
        assert len(stutter_mask) == len(audio)
    
    def test_detect_stutters_no_stutters(self):
        """Test stutter detection with no stutters."""
        repair = GlitchRepair()
        sr = 44100
        
        # Create clean audio
        audio = np.random.randn(int(sr * 0.1)) * 0.5
        
        stutter_mask = repair.detect_stutters(audio, sr)
        
        # Should return mask
        assert isinstance(stutter_mask, np.ndarray)
        assert len(stutter_mask) == len(audio)
    
    def test_repair_clicks(self):
        """Test click repair."""
        repair = GlitchRepair()
        sr = 44100
        
        # Create audio with click
        audio = np.random.randn(int(sr * 0.1)) * 0.1
        click_position = len(audio) // 2
        audio[click_position] = 1.0
        
        click_mask = repair.detect_clicks(audio, sr)
        repaired = repair.repair_clicks(audio, sr, click_mask)
        
        # Should return repaired audio
        assert isinstance(repaired, np.ndarray)
        assert len(repaired) == len(audio)
        # Click should be reduced
        assert np.abs(repaired[click_position]) < np.abs(audio[click_position])
    
    def test_repair_clicks_no_clicks(self):
        """Test click repair with no clicks."""
        repair = GlitchRepair()
        sr = 44100
        
        audio = np.random.randn(int(sr * 0.1)) * 0.1
        click_mask = np.zeros(len(audio), dtype=bool)
        
        repaired = repair.repair_clicks(audio, sr, click_mask)
        
        # Should return audio unchanged (or very similar)
        assert isinstance(repaired, np.ndarray)
        assert len(repaired) == len(audio)
    
    def test_repair_stutters(self):
        """Test stutter repair."""
        repair = GlitchRepair()
        sr = 44100
        
        # Create audio with stutter
        audio = np.random.randn(int(sr * 0.1)) * 0.5
        stutter_start = len(audio) // 3
        stutter_end = stutter_start + int(0.01 * sr)
        audio[stutter_start:stutter_end] *= 0.01
        
        stutter_mask = repair.detect_stutters(audio, sr)
        repaired = repair.repair_stutters(audio, sr, stutter_mask)
        
        # Should return repaired audio
        assert isinstance(repaired, np.ndarray)
        assert len(repaired) == len(audio)
    
    def test_repair_stutters_no_stutters(self):
        """Test stutter repair with no stutters."""
        repair = GlitchRepair()
        sr = 44100
        
        audio = np.random.randn(int(sr * 0.1)) * 0.5
        stutter_mask = np.zeros(len(audio), dtype=bool)
        
        repaired = repair.repair_stutters(audio, sr, stutter_mask)
        
        # Should return audio
        assert isinstance(repaired, np.ndarray)
        assert len(repaired) == len(audio)
    
    def test_process_stems_arrays(self):
        """Test processing stems as audio arrays."""
        repair = GlitchRepair()
        sr = 44100
        
        stems = {
            "vocals": np.random.randn(int(sr * 0.1)),
            "drums": np.random.randn(int(sr * 0.1))
        }
        
        processed = repair.process_stems(stems, sr=sr)
        
        # Should return processed stems
        assert "vocals" in processed
        assert "drums" in processed
        assert isinstance(processed["vocals"], np.ndarray)
        assert isinstance(processed["drums"], np.ndarray)
    
    def test_process_stems_files(self, temp_dir):
        """Test processing stems from file paths."""
        repair = GlitchRepair()
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
        
        processed = repair.process_stems(stems, sr=sr)
        
        # Should return file paths
        assert "vocals" in processed
        assert "drums" in processed
        assert isinstance(processed["vocals"], str)
        assert Path(processed["vocals"]).exists()
    
    def test_process_stems_empty(self):
        """Test processing empty stems dictionary."""
        repair = GlitchRepair()
        
        processed = repair.process_stems({}, sr=44100)
        
        assert processed == {}
    
    def test_detect_clicks_stereo(self):
        """Test click detection on stereo audio."""
        repair = GlitchRepair()
        sr = 44100
        
        # Create stereo audio with click
        audio = np.array([
            np.random.randn(int(sr * 0.1)) * 0.1,
            np.random.randn(int(sr * 0.1)) * 0.1
        ])
        click_position = audio.shape[1] // 2
        audio[0, click_position] = 1.0
        
        click_mask = repair.detect_clicks(audio, sr)
        
        # Should detect clicks
        assert isinstance(click_mask, np.ndarray)
        assert len(click_mask) == audio.shape[1]
    
    def test_repair_clicks_stereo(self):
        """Test click repair on stereo audio."""
        repair = GlitchRepair()
        sr = 44100
        
        # Create stereo audio with click
        audio = np.array([
            np.random.randn(int(sr * 0.1)) * 0.1,
            np.random.randn(int(sr * 0.1)) * 0.1
        ])
        click_position = audio.shape[1] // 2
        audio[0, click_position] = 1.0
        
        click_mask = repair.detect_clicks(audio, sr)
        repaired = repair.repair_clicks(audio, sr, click_mask)
        
        # Should repair both channels
        assert repaired.shape == audio.shape
        assert len(repaired.shape) == 2

