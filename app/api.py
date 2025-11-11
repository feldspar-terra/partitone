"""REST API for PartiTone."""

import os
import tempfile
import zipfile
import logging
import json
import threading
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.runner import DemucsRunner, get_device_info
from app.utils import is_audio_file, ensure_directory
from app.background import job_manager, JobStatus
from app.config import DEFAULT_PRESET, PERFORMANCE_PRESETS, MAX_FILE_SIZE_MB, SSE_POLL_INTERVAL, get_preset
from app.preset_utils import apply_preset
from app.exceptions import (
    PartiToneError,
    InvalidAudioFormatError,
    FileSizeLimitError,
    FileReadError,
    ModelInitializationError,
    JobNotFoundError,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PartiTone API",
    description="Audio stem separation API using Demucs",
    version="1.0.0",
)

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    gpu_available: bool
    device_info: dict


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    device_info = get_device_info()
    return HealthResponse(
        status="healthy",
        gpu_available=device_info['cuda_available'],
        device_info=device_info,
    )


@app.get("/presets")
async def get_presets():
    """Get available performance presets."""
    presets_info = {}
    for name, preset in PERFORMANCE_PRESETS.items():
        presets_info[name] = {
            "name": preset.name,
            "model": preset.model,
            "raw": preset.raw,
            "repair_glitch": preset.repair_glitch,
            "repair_mode": preset.repair_mode,
            "description": preset.description,
            "estimated_time_3min_song": preset.estimated_time_3min_song,
        }
    return {
        "presets": presets_info,
        "default": DEFAULT_PRESET,
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI."""
    index_path = static_path / "index.html"
    if index_path.exists():
        with open(index_path, 'r') as f:
            return f.read()
    return {
        "name": "PartiTone API",
        "version": "1.0.0",
        "endpoints": {
            "POST /separate": "Separate audio file into stems",
            "GET /health": "Health check",
            "GET /progress/{job_id}": "SSE endpoint for job progress",
            "GET /download/{job_id}": "Download processed stems",
        },
    }


@app.post("/separate")
async def separate_audio(
    file: UploadFile = File(...),
    stems: str = Form(None),
    model: str = Form(None),
    format: str = Form("wav"),
    device: Optional[str] = Form(None),
    raw: bool = Form(None),
    repair_glitch: bool = Form(None),
    preset: str = Form(None),
    repair_mode: str = Form(None),
):
    """Separate audio file into stems (web UI version with background processing).
    
    This is the recommended endpoint for new integrations. It provides:
    - Background job processing (non-blocking)
    - Real-time progress updates via SSE
    - Detailed processing logs
    - Performance metrics
    
    Args:
        file: Audio file to separate (MP3, WAV, FLAC, or M4A)
        stems: JSON array of selected stems (e.g., ["vocals", "drums"])
        model: Model to use (default: from preset or htdemucs)
        format: Output format - wav or flac (default: wav)
        device: Device to use - cpu, cuda, or auto (default: auto, selects best GPU on multi-GPU systems)
        raw: Skip light cleanup (default: from preset or False)
        repair_glitch: Enable glitch repair (default: from preset or False)
        preset: Performance preset - fast, balanced, quality, ultra (overrides individual settings)
        repair_mode: Glitch repair mode - fast, standard, thorough (default: standard)
    
    Returns:
        JSON response with job_id for tracking progress:
        ```json
        {
            "job_id": "uuid-here",
            "status": "queued"
        }
        ```
    
    Raises:
        HTTPException: 400 if file format is invalid or preset is invalid
        HTTPException: 413 if file size exceeds maximum allowed size
        HTTPException: 500 if file reading fails
    """
    # Validate file type
    filename = file.filename
    if not filename or not is_audio_file(filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload MP3, WAV, FLAC, M4A, or AAC file."
        )
    
    # Read file content and validate size
    try:
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        logger.info(f"Received file: {filename} ({len(file_content)} bytes, {file_size_mb:.2f} MB)")
        
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB} MB)"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to process uploaded file")
    
    # Parse selected stems
    selected_stems = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']
    if stems:
        try:
            selected_stems = json.loads(stems)
            # Validate stems
            valid_stems = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']
            selected_stems = [s for s in selected_stems if s in valid_stems]
            if not selected_stems:
                raise HTTPException(
                    status_code=400,
                    detail="At least one valid stem must be selected"
                )
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid stems parameter. Expected JSON array."
            )
    
    # Validate format
    format = format.lower()
    if format not in ['wav', 'flac']:
        format = 'wav'
    
    # Determine device
    device = None if device == 'auto' or device is None else device
    
    # Apply preset if provided (preset overrides individual settings)
    try:
        model, raw, repair_glitch, repair_mode = apply_preset(preset)
        if preset:
            preset_config = get_preset(preset)
            logger.info(f"Using preset '{preset}': {preset_config.description}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Override with explicit parameters if provided (when no preset)
    if not preset:
        if model is None:
            model = "htdemucs"
        if raw is None:
            raw = False
        if repair_glitch is None:
            repair_glitch = False
        if repair_mode is None:
            repair_mode = "standard"
    
    # Create job
    job_id = job_manager.create_job(
        filename=filename,
        selected_stems=selected_stems,
        format=format,
        raw=raw,
        repair_glitch=repair_glitch,
        repair_mode=repair_mode,
        model=model
    )
    
    # Process job in background thread
    def process_in_background():
        job_manager.process_job(job_id, file_content, model=model, device=device)
    
    thread = threading.Thread(target=process_in_background, daemon=True)
    thread.start()
    
    return {"job_id": job_id, "status": "queued"}


@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    """Server-Sent Events endpoint for job progress updates."""
    async def event_generator():
        job = job_manager.get_job(job_id)
        if not job:
            yield f"data: {json.dumps({'status': 'failed', 'message': 'Job not found'})}\n\n"
            return
        
        last_status = None
        while True:
            job = job_manager.get_job(job_id)
            if not job:
                yield f"data: {json.dumps({'status': 'failed', 'message': 'Job not found'})}\n\n"
                break
            
            # Send update if status changed
            if job.status != last_status or job.status == JobStatus.PROCESSING:
                data = {
                    'status': job.status,
                    'message': job.message,
                    'progress': job.progress,
                    'log_messages': job.log_messages,  # Include full log history
                }
                if job.error:
                    data['error'] = job.error
                
                yield f"data: {json.dumps(data)}\n\n"
                
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    break
                
                last_status = job.status
            
            # Small delay to avoid overwhelming the client
            import asyncio
            await asyncio.sleep(SSE_POLL_INTERVAL)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/download/{job_id}")
async def download_stems(job_id: str):
    """Download processed stems ZIP file."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job.status}"
        )
    
    if not job.zip_path or not job.zip_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=str(job.zip_path),
        media_type="application/zip",
        filename=f"{Path(job.filename).stem}_stems.zip",
    )


@app.post("/separate-legacy")
async def separate_audio_legacy(
    file: UploadFile = File(...),
    model: str = "htdemucs",
    format: str = "wav",
    device: Optional[str] = None,
    raw: bool = False,
    repair_glitch: bool = False,
):
    """Legacy separate endpoint (synchronous, returns ZIP directly).
    
    **DEPRECATED**: This endpoint is kept for backward compatibility with older API clients.
    For new integrations, use `/separate` which provides background processing and progress tracking.
    
    **Differences from `/separate`:**
    - Synchronous processing (blocks until complete)
    - Returns ZIP file directly in response (no job tracking)
    - No progress updates or log messages
    - Simpler for simple scripts but less efficient for large files
    
    **Request:**
    - Content-Type: `multipart/form-data`
    - Field name: `file`
    - Accepted formats: MP3, WAV, FLAC, M4A
    - Form fields:
      - `model` (string, optional): Model to use (default: "htdemucs")
      - `format` (string, optional): Output format - wav or flac (default: "wav")
      - `device` (string, optional): Device to use - cpu, cuda, or auto (default: auto)
      - `raw` (bool, optional): Skip cleanup (default: False)
      - `repair_glitch` (bool, optional): Enable glitch repair (default: False)
    
    **Response:**
    - Content-Type: `application/zip`
    - ZIP file containing all separated stems
    
    **Example:**
    ```bash
    curl -F "file=@song.mp3" \
      -F "model=htdemucs" \
      -F "format=wav" \
      http://localhost:8000/separate-legacy \
      --output stems.zip
    ```
    """
    # Validate file type
    filename = file.filename
    if not filename or not is_audio_file(filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload MP3, WAV, or FLAC file."
        )
    
    # Validate format
    format = format.lower()
    if format not in ['wav', 'flac']:
        format = 'wav'
    
    # Determine device
    device = None if device == 'auto' or device is None else device
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / filename
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir()
        
        # Validate file size
        try:
            content = await file.read()
            file_size_mb = len(content) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                raise HTTPException(
                    status_code=413,
                    detail=f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB} MB)"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to read uploaded file: {e}")
            raise HTTPException(status_code=500, detail="Failed to process uploaded file")
        
        # Save uploaded file
        try:
            with open(input_path, 'wb') as f:
                f.write(content)
            logger.info(f"Saved uploaded file: {filename} ({len(content)} bytes)")
        except (IOError, OSError) as e:
            logger.error(f"Failed to save uploaded file: {e}")
            raise HTTPException(status_code=500, detail="Failed to process uploaded file")
        
        # Initialize runner
        try:
            runner = DemucsRunner(model=model, device=device)
        except Exception as e:
            logger.error(f"Failed to initialize Demucs runner: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")
        
        # Separate audio
        try:
            result = runner.separate(
                str(input_path), 
                str(output_dir), 
                format,
                raw=raw,
                repair_glitch=repair_glitch
            )
            logger.info(f"Separation complete: {result['output_dir']}")
        except (RuntimeError, FileNotFoundError, ValueError) as e:
            logger.error(f"Separation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Separation failed: {str(e)}")
        
        # Create ZIP archive
        zip_path = Path(temp_dir) / "stems.zip"
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for stem_name, stem_path in result['stems'].items():
                    if os.path.exists(stem_path):
                        arcname = f"{Path(stem_path).name}"
                        zipf.write(stem_path, arcname=arcname)
                        logger.info(f"Added {stem_name} to ZIP: {arcname}")
            
            logger.info(f"Created ZIP archive: {zip_path}")
        except (zipfile.BadZipFile, IOError, OSError) as e:
            logger.error(f"Failed to create ZIP archive: {e}")
            raise HTTPException(status_code=500, detail="Failed to create output archive")
        
        # Return ZIP file
        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename=f"{Path(filename).stem}_stems.zip",
        )

