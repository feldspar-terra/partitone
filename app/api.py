"""REST API for PartiTone."""

import os
import tempfile
import zipfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.runner import DemucsRunner, get_device_info
from app.utils import is_audio_file, ensure_directory


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PartiTone API",
    description="Audio stem separation API using Demucs",
    version="1.0.0",
)


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


@app.post("/separate")
async def separate_audio(
    file: UploadFile = File(...),
    model: str = "htdemucs",
    format: str = "wav",
    device: Optional[str] = None,
    raw: bool = False,
    repair_glitch: bool = False,
):
    """Separate audio file into stems.
    
    Args:
        file: Audio file to separate (MP3, WAV, FLAC, or M4A)
        model: Model to use (default: htdemucs)
        format: Output format - wav or flac (default: wav)
        device: Device to use - cpu, cuda, or auto (default: auto)
        raw: Skip light cleanup (default: False)
        repair_glitch: Enable glitch repair (default: False)
    
    Returns:
        ZIP file containing separated stems
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
        
        # Save uploaded file
        try:
            with open(input_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            logger.info(f"Saved uploaded file: {filename} ({len(content)} bytes)")
        except Exception as e:
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
        except Exception as e:
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
        except Exception as e:
            logger.error(f"Failed to create ZIP archive: {e}")
            raise HTTPException(status_code=500, detail="Failed to create output archive")
        
        # Return ZIP file
        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename=f"{Path(filename).stem}_stems.zip",
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "PartiTone API",
        "version": "1.0.0",
        "endpoints": {
            "POST /separate": "Separate audio file into stems",
            "GET /health": "Health check",
        },
    }

