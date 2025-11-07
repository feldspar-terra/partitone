"""Background job processing for PartiTone."""

import os
import uuid
import threading
import logging
import tempfile
import zipfile
import shutil
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from queue import Queue

from app.runner import DemucsRunner
from app.utils import is_audio_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobStatus:
    """Job status constants."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    """Represents a processing job."""
    
    def __init__(self, job_id: str, filename: str, selected_stems: List[str], 
                 format: str = "wav", raw: bool = False, repair_glitch: bool = False):
        self.job_id = job_id
        self.filename = filename
        self.selected_stems = selected_stems
        self.format = format
        self.raw = raw
        self.repair_glitch = repair_glitch
        self.status = JobStatus.QUEUED
        self.message = "Job queued"
        self.progress = 0
        self.zip_path: Optional[Path] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now()
        self.temp_dir: Optional[Path] = None
        self.log_messages: List[dict] = []  # List of log entries with timestamp and message


class JobManager:
    """Manages background processing jobs."""
    
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.lock = threading.Lock()
        self.cleanup_interval = timedelta(hours=1)  # Clean up jobs older than 1 hour
    
    def create_job(self, filename: str, selected_stems: List[str], 
                   format: str = "wav", raw: bool = False, 
                   repair_glitch: bool = False) -> str:
        """Create a new job and return job ID."""
        job_id = str(uuid.uuid4())
        job = Job(job_id, filename, selected_stems, format, raw, repair_glitch)
        
        with self.lock:
            self.jobs[job_id] = job
        
        logger.info(f"Created job {job_id} for {filename}")
        # Add initial log message
        job.log_messages.append({
            'timestamp': datetime.now().isoformat(),
            'message': f"Job created for {filename}",
            'status': JobStatus.QUEUED,
            'progress': 0
        })
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        with self.lock:
            return self.jobs.get(job_id)
    
    def update_job(self, job_id: str, status: str = None, 
                   message: str = None, progress: int = None, 
                   error: str = None, zip_path: Path = None):
        """Update job status."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            
            if status:
                job.status = status
            if message:
                job.message = message
                # Add message to log with timestamp
                job.log_messages.append({
                    'timestamp': datetime.now().isoformat(),
                    'message': message,
                    'status': status or job.status,
                    'progress': progress if progress is not None else job.progress
                })
            if progress is not None:
                job.progress = progress
            if error:
                job.error = error
                # Add error to log
                job.log_messages.append({
                    'timestamp': datetime.now().isoformat(),
                    'message': f"Error: {error}",
                    'status': status or job.status,
                    'progress': progress if progress is not None else job.progress
                })
            if zip_path:
                job.zip_path = zip_path
    
    def process_job(self, job_id: str, file_content: bytes, 
                    model: str = "htdemucs", device: Optional[str] = None):
        """Process a job in the background."""
        job = self.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        
        try:
            # Create temporary directory for this job
            temp_dir = Path(tempfile.mkdtemp(prefix=f"partitone_{job_id}_"))
            job.temp_dir = temp_dir
            
            self.update_job(job_id, status=JobStatus.PROCESSING, 
                           message="Saving uploaded file...", progress=5)
            
            # Save uploaded file
            input_path = temp_dir / job.filename
            with open(input_path, 'wb') as f:
                f.write(file_content)
            
            self.update_job(job_id, message="Initializing model...", progress=10)
            
            # Initialize runner
            runner = DemucsRunner(model=model, device=device)
            
            output_dir = temp_dir / "output"
            output_dir.mkdir()
            
            self.update_job(job_id, message="Separating audio stems...", progress=20)
            
            # Separate audio
            result = runner.separate(
                str(input_path),
                str(output_dir),
                job.format,
                raw=job.raw,
                repair_glitch=job.repair_glitch
            )
            
            self.update_job(job_id, message="Creating ZIP archive...", progress=80)
            
            # Create ZIP with only selected stems
            zip_path = temp_dir / "stems.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for stem_name in job.selected_stems:
                    if stem_name in result['stems']:
                        stem_path = result['stems'][stem_name]
                        if os.path.exists(stem_path):
                            arcname = f"{Path(stem_path).name}"
                            zipf.write(stem_path, arcname=arcname)
                            logger.info(f"Added {stem_name} to ZIP: {arcname}")
            
            self.update_job(job_id, 
                           status=JobStatus.COMPLETED,
                           message="Processing complete!",
                           progress=100,
                           zip_path=zip_path)
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            self.update_job(job_id,
                           status=JobStatus.FAILED,
                           message=f"Processing failed: {str(e)}",
                           error=str(e))
    
    def cleanup_old_jobs(self):
        """Clean up old completed/failed jobs."""
        now = datetime.now()
        with self.lock:
            to_remove = []
            for job_id, job in self.jobs.items():
                if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED] and
                    now - job.created_at > self.cleanup_interval):
                    to_remove.append(job_id)
            
            for job_id in to_remove:
                job = self.jobs[job_id]
                # Clean up temporary directory
                if job.temp_dir and job.temp_dir.exists():
                    try:
                        shutil.rmtree(job.temp_dir)
                        logger.info(f"Cleaned up temp directory for job {job_id}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up job {job_id}: {e}")
                del self.jobs[job_id]
                logger.info(f"Removed old job {job_id}")


# Global job manager instance
job_manager = JobManager()

