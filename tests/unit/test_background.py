"""Unit tests for background job processing."""

import pytest
from pathlib import Path
from datetime import datetime, timedelta

from app.background import Job, JobManager, JobStatus


class TestJob:
    """Tests for Job class."""
    
    def test_job_initialization(self):
        """Test that Job is initialized correctly."""
        job = Job(
            job_id="test-id",
            filename="test.mp3",
            selected_stems=["vocals", "drums"],
            format="wav",
            raw=False,
            repair_glitch=False
        )
        
        assert job.job_id == "test-id"
        assert job.filename == "test.mp3"
        assert job.selected_stems == ["vocals", "drums"]
        assert job.format == "wav"
        assert job.raw is False
        assert job.repair_glitch is False
        assert job.status == JobStatus.QUEUED
        assert job.message == "Job queued"
        assert job.progress == 0
        assert job.zip_path is None
        assert job.error is None
        assert isinstance(job.created_at, datetime)
        assert job.temp_dir is None
        assert isinstance(job.log_messages, list)
        assert len(job.log_messages) == 0
    
    def test_job_with_options(self):
        """Test Job initialization with all options."""
        job = Job(
            job_id="test-id",
            filename="test.flac",
            selected_stems=["vocals"],
            format="flac",
            raw=True,
            repair_glitch=True
        )
        
        assert job.format == "flac"
        assert job.raw is True
        assert job.repair_glitch is True


class TestJobManager:
    """Tests for JobManager class."""
    
    def test_create_job(self):
        """Test job creation."""
        manager = JobManager()
        job_id = manager.create_job(
            filename="test.mp3",
            selected_stems=["vocals", "drums"]
        )
        
        assert job_id is not None
        assert len(job_id) > 0
        
        job = manager.get_job(job_id)
        assert job is not None
        assert job.filename == "test.mp3"
        assert job.selected_stems == ["vocals", "drums"]
        assert job.status == JobStatus.QUEUED
        assert len(job.log_messages) == 1
        assert "Job created" in job.log_messages[0]["message"]
    
    def test_get_job(self):
        """Test job retrieval."""
        manager = JobManager()
        job_id = manager.create_job("test.mp3", ["vocals"])
        
        job = manager.get_job(job_id)
        assert job is not None
        assert job.job_id == job_id
        
        # Test nonexistent job
        assert manager.get_job("nonexistent") is None
    
    def test_update_job_status(self):
        """Test updating job status."""
        manager = JobManager()
        job_id = manager.create_job("test.mp3", ["vocals"])
        
        manager.update_job(job_id, status=JobStatus.PROCESSING)
        job = manager.get_job(job_id)
        assert job.status == JobStatus.PROCESSING
    
    def test_update_job_message(self):
        """Test updating job message and log tracking."""
        manager = JobManager()
        job_id = manager.create_job("test.mp3", ["vocals"])
        
        initial_log_count = len(manager.get_job(job_id).log_messages)
        
        manager.update_job(job_id, message="Processing started")
        job = manager.get_job(job_id)
        
        assert job.message == "Processing started"
        assert len(job.log_messages) == initial_log_count + 1
        assert job.log_messages[-1]["message"] == "Processing started"
        assert "timestamp" in job.log_messages[-1]
        assert "status" in job.log_messages[-1]
        assert "progress" in job.log_messages[-1]
    
    def test_update_job_progress(self):
        """Test updating job progress."""
        manager = JobManager()
        job_id = manager.create_job("test.mp3", ["vocals"])
        
        manager.update_job(job_id, progress=50)
        job = manager.get_job(job_id)
        assert job.progress == 50
    
    def test_update_job_error(self):
        """Test updating job error and log tracking."""
        manager = JobManager()
        job_id = manager.create_job("test.mp3", ["vocals"])
        
        initial_log_count = len(manager.get_job(job_id).log_messages)
        
        manager.update_job(job_id, error="Test error")
        job = manager.get_job(job_id)
        
        assert job.error == "Test error"
        assert len(job.log_messages) == initial_log_count + 1
        assert "Error: Test error" in job.log_messages[-1]["message"]
    
    def test_update_job_zip_path(self):
        """Test updating job ZIP path."""
        manager = JobManager()
        job_id = manager.create_job("test.mp3", ["vocals"])
        
        zip_path = Path("/tmp/test.zip")
        manager.update_job(job_id, zip_path=zip_path)
        job = manager.get_job(job_id)
        assert job.zip_path == zip_path
    
    def test_update_nonexistent_job(self):
        """Test that updating nonexistent job doesn't crash."""
        manager = JobManager()
        # Should not raise exception
        manager.update_job("nonexistent", status=JobStatus.PROCESSING)
    
    def test_multiple_jobs(self):
        """Test managing multiple jobs."""
        manager = JobManager()
        
        job_id1 = manager.create_job("test1.mp3", ["vocals"])
        job_id2 = manager.create_job("test2.mp3", ["drums"])
        
        assert job_id1 != job_id2
        
        job1 = manager.get_job(job_id1)
        job2 = manager.get_job(job_id2)
        
        assert job1.filename == "test1.mp3"
        assert job2.filename == "test2.mp3"
    
    def test_job_status_transitions(self):
        """Test job status transitions."""
        manager = JobManager()
        job_id = manager.create_job("test.mp3", ["vocals"])
        
        # Queued -> Processing
        manager.update_job(job_id, status=JobStatus.PROCESSING)
        assert manager.get_job(job_id).status == JobStatus.PROCESSING
        
        # Processing -> Completed
        manager.update_job(job_id, status=JobStatus.COMPLETED)
        assert manager.get_job(job_id).status == JobStatus.COMPLETED
    
    def test_log_messages_accumulation(self):
        """Test that log messages accumulate correctly."""
        manager = JobManager()
        job_id = manager.create_job("test.mp3", ["vocals"])
        
        # Initial log message from creation
        assert len(manager.get_job(job_id).log_messages) == 1
        
        # Add multiple messages
        manager.update_job(job_id, message="Step 1", progress=10)
        manager.update_job(job_id, message="Step 2", progress=20)
        manager.update_job(job_id, message="Step 3", progress=30)
        
        job = manager.get_job(job_id)
        assert len(job.log_messages) == 4  # 1 initial + 3 updates
        
        # Check that messages are in order
        assert "Job created" in job.log_messages[0]["message"]
        assert job.log_messages[1]["message"] == "Step 1"
        assert job.log_messages[2]["message"] == "Step 2"
        assert job.log_messages[3]["message"] == "Step 3"
    
    def test_log_message_timestamps(self):
        """Test that log messages have timestamps."""
        manager = JobManager()
        job_id = manager.create_job("test.mp3", ["vocals"])
        
        manager.update_job(job_id, message="Test message")
        job = manager.get_job(job_id)
        
        # Check timestamp format (ISO format)
        timestamp = job.log_messages[-1]["timestamp"]
        assert isinstance(timestamp, str)
        # Should be parseable as ISO format
        parsed = datetime.fromisoformat(timestamp)
        assert isinstance(parsed, datetime)
    
    def test_log_message_with_status_and_progress(self):
        """Test that log messages include status and progress."""
        manager = JobManager()
        job_id = manager.create_job("test.mp3", ["vocals"])
        
        manager.update_job(
            job_id,
            status=JobStatus.PROCESSING,
            message="Processing",
            progress=50
        )
        
        job = manager.get_job(job_id)
        log_entry = job.log_messages[-1]
        
        assert log_entry["status"] == JobStatus.PROCESSING
        assert log_entry["progress"] == 50
        assert log_entry["message"] == "Processing"

