"""Unit tests for DemucsRunner class."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.runner import DemucsRunner, get_device_info


class TestDemucsRunner:
    """Tests for DemucsRunner class."""
    
    def test_initialization_default(self):
        """Test DemucsRunner initialization with defaults."""
        runner = DemucsRunner()
        assert runner.model_name == "htdemucs"
        assert runner.device in ["cpu", "cuda"]
    
    def test_initialization_with_model(self):
        """Test DemucsRunner initialization with custom model."""
        runner = DemucsRunner(model="htdemucs_ft")
        assert runner.model_name == "htdemucs_ft"
    
    @patch('app.runner.torch.cuda.is_available')
    def test_initialization_auto_device_cpu(self, mock_cuda_available):
        """Test auto device detection when CUDA not available."""
        mock_cuda_available.return_value = False
        runner = DemucsRunner(device=None)
        assert runner.device == "cpu"
    
    @patch('app.runner.torch.cuda.is_available')
    def test_initialization_auto_device_cuda(self, mock_cuda_available):
        """Test auto device detection when CUDA is available."""
        mock_cuda_available.return_value = True
        runner = DemucsRunner(device=None)
        assert runner.device == "cuda"
    
    def test_initialization_explicit_device(self):
        """Test explicit device specification."""
        runner = DemucsRunner(device="cpu")
        assert runner.device == "cpu"
        
        # Note: cuda device test would require actual CUDA, so we skip it
    
    def test_separate_file_not_found(self, temp_dir):
        """Test that separate raises error for nonexistent file."""
        runner = DemucsRunner()
        nonexistent_file = temp_dir / "nonexistent.wav"
        
        with pytest.raises(FileNotFoundError):
            runner.separate(str(nonexistent_file), str(temp_dir))
    
    def test_separate_invalid_format(self, temp_dir):
        """Test that separate raises error for invalid audio format."""
        runner = DemucsRunner()
        invalid_file = temp_dir / "test.txt"
        invalid_file.write_text("not audio")
        
        with pytest.raises(ValueError, match="Unsupported audio format"):
            runner.separate(str(invalid_file), str(temp_dir))
    
    @pytest.mark.skip(reason="Requires Demucs and actual audio processing")
    def test_separate_success(self):
        """Test successful separation (skipped - requires Demucs)."""
        pass
    
    def test_separate_batch_empty_directory(self, temp_dir):
        """Test batch processing with empty directory."""
        runner = DemucsRunner()
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        results = runner.separate_batch(str(empty_dir), str(temp_dir))
        assert results == []
    
    @pytest.mark.skip(reason="Requires Demucs and actual audio processing")
    def test_separate_batch_with_files(self):
        """Test batch processing with files (skipped - requires Demucs)."""
        pass


class TestGetDeviceInfo:
    """Tests for get_device_info function."""
    
    @patch('app.runner.torch.cuda.is_available')
    @patch('app.runner.torch.cuda.device_count')
    @patch('app.runner.torch.cuda.get_device_name')
    def test_get_device_info_cuda_available(self, mock_get_name, mock_count, mock_available):
        """Test device info when CUDA is available."""
        mock_available.return_value = True
        mock_count.return_value = 1
        mock_get_name.return_value = "NVIDIA GeForce RTX 3080"
        
        info = get_device_info()
        
        assert info["cuda_available"] is True
        assert info["device_count"] == 1
        assert info["device_name"] == "NVIDIA GeForce RTX 3080"
    
    @patch('app.runner.torch.cuda.is_available')
    def test_get_device_info_cuda_unavailable(self, mock_available):
        """Test device info when CUDA is not available."""
        mock_available.return_value = False
        
        info = get_device_info()
        
        assert info["cuda_available"] is False
        assert info["device_count"] == 0
        assert info["device_name"] is None

