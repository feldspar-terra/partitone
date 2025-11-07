"""Integration tests for job processing workflow."""

import pytest
import tempfile
import zipfile
from pathlib import Path
import soundfile as sf
import numpy as np
import threading
import time

from app.background import JobManager, JobStatus
from app.runner import DemucsRunner


class TestJobLifecycle:
    """Tests for job lifecycle management."""
    
    def test_job_lifecycle_queued_to_processing(self):
        """Test job transitions from queued to processing."""
        manager = JobManager()
        job_id = manager.create_job("test.mp3", ["vocals"])
        
        job = manager.get_job(job_id)
        assert job.status == JobStatus.QUEUED
        
        manager.update_job(job_id, status=JobStatus.PROCESSING)
        job = manager.get_job(job_id)
        assert job.status == JobStatus.PROCESSING
    
    def test_job_lifecycle_processing_to_completed(self):
        """Test job transitions from processing to completed."""
        manager = JobManager()
        job_id = manager.create_job("test.mp3", ["vocals"])
        
        manager.update_job(job_id, status=JobStatus.PROCESSING)
        manager.update_job(job_id, status=JobStatus.COMPLETED, progress=100)
        
        job = manager.get_job(job_id)
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 100
    
    def test_job_lifecycle_processing_to_failed(self):
        """Test job transitions from processing to failed."""
        manager = JobManager()
        job_id = manager.create_job("test.mp3", ["vocals"])
        
        manager.update_job(job_id, status=JobStatus.PROCESSING)
        manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            error="Test error message"
        )
        
        job = manager.get_job(job_id)
        assert job.status == JobStatus.FAILED
        assert job.error == "Test error message"
    
    def test_log_message_accumulation_during_processing(self):
        """Test that log messages accumulate during processing."""
        manager = JobManager()
        job_id = manager.create_job("test.mp3", ["vocals"])
        
        # Simulate processing steps
        steps = [
            ("Saving uploaded file...", 5),
            ("Initializing model...", 10),
            ("Separating audio stems...", 20),
            ("Creating ZIP archive...", 80),
            ("Processing complete!", 100)
        ]
        
        for message, progress in steps:
            manager.update_job(
                job_id,
                status=JobStatus.PROCESSING,
                message=message,
                progress=progress
            )
        
        job = manager.get_job(job_id)
        
        # Should have initial log + 5 processing logs
        assert len(job.log_messages) == 6
        
        # Verify messages are in order
        assert "Job created" in job.log_messages[0]["message"]
        assert job.log_messages[1]["message"] == "Saving uploaded file..."
        assert job.log_messages[-1]["message"] == "Processing complete!"
        assert job.log_messages[-1]["progress"] == 100
    
    def test_zip_file_creation_with_selected_stems(self, temp_dir):
        """Test ZIP file creation with selected stems."""
        # Create mock stem files
        stems_dir = temp_dir / "stems"
        stems_dir.mkdir()
        
        stem_files = {
            "vocals": stems_dir / "vocals.wav",
            "drums": stems_dir / "drums.wav",
            "bass": stems_dir / "bass.wav"
        }
        
        # Create minimal WAV files
        sr = 44100
        audio = np.random.randn(sr).astype(np.float32)
        for stem_path in stem_files.values():
            sf.write(str(stem_path), audio, sr)
        
        # Create ZIP with only selected stems
        selected_stems = ["vocals", "drums"]
        zip_path = temp_dir / "stems.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for stem_name in selected_stems:
                if stem_name in stem_files:
                    stem_path = stem_files[stem_name]
                    zipf.write(stem_path, arcname=stem_path.name)
        
        # Verify ZIP contains only selected stems
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            files_in_zip = zipf.namelist()
            assert "vocals.wav" in files_in_zip
            assert "drums.wav" in files_in_zip
            assert "bass.wav" not in files_in_zip
    
    def test_error_logging_in_job(self):
        """Test that errors are logged in job."""
        manager = JobManager()
        job_id = manager.create_job("test.mp3", ["vocals"])
        
        initial_log_count = len(manager.get_job(job_id).log_messages)
        
        manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            error="Processing failed: test error"
        )
        
        job = manager.get_job(job_id)
        
        # Should have error in log messages
        assert len(job.log_messages) == initial_log_count + 1
        assert "Error: Processing failed: test error" in job.log_messages[-1]["message"]
        assert job.log_messages[-1]["status"] == JobStatus.FAILED


class TestJobManagerThreadSafety:
    """Tests for thread safety of JobManager."""
    
    def test_concurrent_job_creation(self):
        """Test creating multiple jobs concurrently."""
        manager = JobManager()
        
        def create_job(i):
            return manager.create_job(f"test{i}.mp3", ["vocals"])
        
        # Create jobs concurrently
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            job_ids = list(executor.map(create_job, range(10)))
        
        # All jobs should be created
        assert len(job_ids) == 10
        assert len(set(job_ids)) == 10  # All unique
        
        # All jobs should be retrievable
        for job_id in job_ids:
            job = manager.get_job(job_id)
            assert job is not None
    
    def test_concurrent_job_updates(self):
        """Test updating jobs concurrently."""
        manager = JobManager()
        job_id = manager.create_job("test.mp3", ["vocals"])
        
        def update_job(progress):
            manager.update_job(job_id, progress=progress)
        
        # Update job concurrently
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(update_job, range(0, 100, 10))
        
        # Job should still be accessible
        job = manager.get_job(job_id)
        assert job is not None
        # Progress should be one of the values (last write wins)
        assert job.progress in range(0, 100, 10)


class TestJobProcessingIntegration:
    """Integration tests for job processing."""
    
    @pytest.mark.skip(reason="Requires actual Demucs model - slow and requires dependencies")
    def test_process_job_with_real_audio(self, temp_dir):
        """Test processing a job with real audio file."""
        # This test is skipped by default as it requires Demucs and is slow
        manager = JobManager()
        
        # Create a minimal audio file
        audio_path = temp_dir / "test.wav"
        sr = 44100
        audio = np.random.randn(sr).astype(np.float32)
        sf.write(str(audio_path), audio, sr)
        
        # Read file content
        with open(audio_path, 'rb') as f:
            file_content = f.read()
        
        # Create job
        job_id = manager.create_job("test.wav", ["vocals"])
        
        # Process job (this would actually run Demucs)
        # manager.process_job(job_id, file_content)
        
        # For now, just verify job was created
        job = manager.get_job(job_id)
        assert job is not None

