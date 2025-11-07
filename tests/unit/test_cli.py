"""Unit tests for CLI interface."""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from app.cli import parse_args, main


class TestParseArgs:
    """Tests for parse_args function."""
    
    def test_parse_args_minimal(self):
        """Test parsing minimal arguments."""
        args = parse_args(["test.mp3"])
        assert args.input == "test.mp3"
        assert args.stems == 6
        assert args.model == "htdemucs"
        assert args.output == "/output"
        assert args.format == "wav"
        assert args.batch is False
        assert args.device == "auto"
        assert args.raw is False
        assert args.repair_glitch is False
    
    def test_parse_args_all_options(self):
        """Test parsing all options."""
        args = parse_args([
            "test.mp3",
            "--stems", "4",
            "--model", "htdemucs_ft",
            "--output", "/custom/output",
            "--format", "flac",
            "--batch",
            "--device", "cpu",
            "--raw",
            "--repair-glitch"
        ])
        
        assert args.input == "test.mp3"
        assert args.stems == 4
        assert args.model == "htdemucs_ft"
        assert args.output == "/custom/output"
        assert args.format == "flac"
        assert args.batch is True
        assert args.device == "cpu"
        assert args.raw is True
        assert args.repair_glitch is True
    
    def test_parse_args_batch_flag(self):
        """Test batch flag."""
        args = parse_args(["test.mp3", "--batch"])
        assert args.batch is True
    
    def test_parse_args_raw_flag(self):
        """Test raw flag."""
        args = parse_args(["test.mp3", "--raw"])
        assert args.raw is True
    
    def test_parse_args_repair_glitch_flag(self):
        """Test repair-glitch flag."""
        args = parse_args(["test.mp3", "--repair-glitch"])
        assert args.repair_glitch is True


class TestMain:
    """Tests for main function."""
    
    @patch('app.cli.DemucsRunner')
    @patch('app.cli.validate_input_path')
    @patch('app.cli.is_audio_file')
    @patch('app.cli.ensure_directory')
    def test_main_single_file_success(self, mock_ensure, mock_is_audio, mock_validate, mock_runner_class):
        """Test main function with successful single file processing."""
        mock_validate.return_value = True
        mock_is_audio.return_value = True
        
        mock_runner = MagicMock()
        mock_runner.separate.return_value = {"output_dir": "/output/test"}
        mock_runner_class.return_value = mock_runner
        
        with patch('sys.argv', ['cli.py', 'test.mp3']):
            main()
        
        mock_runner.separate.assert_called_once()
        mock_ensure.assert_called_once()
    
    @patch('app.cli.validate_input_path')
    def test_main_invalid_input_path(self, mock_validate):
        """Test main function with invalid input path."""
        mock_validate.return_value = False
        
        with patch('sys.argv', ['cli.py', 'nonexistent.mp3']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
    
    @patch('app.cli.DemucsRunner')
    @patch('app.cli.validate_input_path')
    @patch('app.cli.is_audio_file')
    def test_main_invalid_audio_format(self, mock_is_audio, mock_validate, mock_runner_class):
        """Test main function with invalid audio format."""
        mock_validate.return_value = True
        mock_is_audio.return_value = False
        
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner
        
        with patch('sys.argv', ['cli.py', 'test.txt']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
    
    @patch('app.cli.DemucsRunner')
    @patch('app.cli.validate_input_path')
    def test_main_runner_initialization_failure(self, mock_validate, mock_runner_class):
        """Test main function when runner initialization fails."""
        mock_validate.return_value = True
        mock_runner_class.side_effect = Exception("Failed to initialize")
        
        with patch('sys.argv', ['cli.py', 'test.mp3']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
    
    @patch('app.cli.DemucsRunner')
    @patch('app.cli.validate_input_path')
    @patch('app.cli.get_audio_files')
    @patch('app.cli.ensure_directory')
    def test_main_batch_mode(self, mock_ensure, mock_get_files, mock_validate, mock_runner_class):
        """Test main function in batch mode."""
        mock_validate.return_value = True
        
        temp_dir = Path(tempfile.mkdtemp())
        mock_get_files.return_value = [str(temp_dir / "file1.mp3"), str(temp_dir / "file2.mp3")]
        
        mock_runner = MagicMock()
        mock_runner.separate_batch.return_value = [
            {"output_dir": "/output/file1"},
            {"output_dir": "/output/file2"}
        ]
        mock_runner_class.return_value = mock_runner
        
        with patch('sys.argv', ['cli.py', str(temp_dir), '--batch']):
            main()
        
        mock_runner.separate_batch.assert_called_once()
    
    @patch('app.cli.DemucsRunner')
    @patch('app.cli.validate_input_path')
    @patch('app.cli.get_audio_files')
    def test_main_batch_mode_no_files(self, mock_get_files, mock_validate, mock_runner_class):
        """Test main function in batch mode with no audio files."""
        mock_validate.return_value = True
        mock_get_files.return_value = []
        
        temp_dir = Path(tempfile.mkdtemp())
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner
        
        with patch('sys.argv', ['cli.py', str(temp_dir), '--batch']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
    
    @patch('app.cli.DemucsRunner')
    @patch('app.cli.validate_input_path')
    @patch('app.cli.is_audio_file')
    def test_main_processing_failure(self, mock_is_audio, mock_validate, mock_runner_class):
        """Test main function when processing fails."""
        mock_validate.return_value = True
        mock_is_audio.return_value = True
        
        mock_runner = MagicMock()
        mock_runner.separate.side_effect = Exception("Processing failed")
        mock_runner_class.return_value = mock_runner
        
        with patch('sys.argv', ['cli.py', 'test.mp3']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

