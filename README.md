# PartiTone

A Dockerized audio stem separation tool that extracts individual stems (vocals, drums, bass, guitar, piano, other) from audio files using Demucs v4. Includes default light cleanup processing to preserve the human element while handling technical issues.

## Features

- **CLI Tool**: Process audio files from the command line
- **REST API**: Separate stems via HTTP API with file upload
- **Web UI**: Modern, user-friendly web interface with real-time progress tracking
- **Processing Log Accordion**: Expandable log viewer showing detailed processing updates with timestamps
- **Performance Presets**: Fast, Balanced, Quality, and Ultra presets for optimized processing
- **GPU Acceleration**: Automatic CUDA detection and GPU acceleration when available (multi-GPU support)
- **Performance Monitoring**: Real-time performance metrics and timing breakdowns
- **Parallel Processing**: Automatic parallelization for batch files and stem processing
- **Batch Processing**: Process entire folders of audio files
- **Multiple Formats**: Supports MP3, WAV, FLAC, and M4A input/output
- **Light Cleanup** (default): Silence trimming, DC offset removal, phase alignment, soft loudness balancing
- **Glitch Repair** (optional): Reduce clicks, pops, stutters, and micro-glitches with fast/standard/thorough modes
- **Dockerized**: All dependencies included, runs anywhere Docker runs

## Requirements

- Docker (Linux, macOS, Windows WSL, Synology, Unraid, UGreen)
- For GPU support: NVIDIA GPU with CUDA-compatible drivers

## Installation

Build the Docker image:

```bash
docker build -t partitone -f docker/Dockerfile .
```

## Usage

### CLI Mode

Process a single audio file:

```bash
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "song.mp3"
```

Process with performance preset:

```bash
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "song.mp3" --preset fast
```

Process with custom options:

```bash
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "song.mp3" --stems 6 --format wav --output /output --preset quality
```

Process a batch folder (with automatic parallel processing):

```bash
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "/input" --batch
```

Batch processing automatically uses parallel processing:
- Multiple files processed simultaneously
- GPU systems: Up to 4 files in parallel
- CPU systems: Uses all CPU cores
- See "Performance Optimization" section for details

#### CLI Options

- `--stems N`: Number of stems to extract (default: 6)
- `--preset PRESET`: Performance preset - fast, balanced, quality, or ultra (overrides --model, --raw, --repair-glitch)
- `--model MODEL`: Model to use (default: "htdemucs", or from preset)
- `--output DIR`: Output directory (default: "/output")
- `--format wav|flac`: Output format (default: "wav")
- `--batch`: Process folder instead of single file
- `--raw`: Skip light cleanup entirely (output raw Demucs stems)
- `--repair-glitch`: Enable glitch repair stage (reduce clicks, pops, stutters)
- `--repair-mode MODE`: Glitch repair mode - fast, standard, or thorough (default: standard)
- `--device DEVICE`: Device to use - cpu, cuda, or auto (default: auto)

### API Mode

Start the API server:

```bash
docker run --rm \
  -p 8000:8000 \
  partitone api
```

Access the web UI at `http://localhost:8000` in your browser.

The web UI provides:
- Drag-and-drop file upload
- Stem selection (vocals, drums, bass, guitar, piano, other)
- **Performance Preset Selection**: Choose from Fast, Balanced, Quality, or Ultra presets
- Real-time progress bar with performance metrics
- **Processing Log Accordion**: Expandable log viewer showing detailed processing updates with timestamps and status indicators
- Direct download of processed stems as ZIP

Separate stems via API (programmatic access):

```bash
curl -F "file=@song.mp3" \
  http://localhost:8000/separate \
  --output stems.zip
```

With GPU support:

```bash
docker run --rm \
  -p 8000:8000 \
  --gpus all \
  partitone api
```

## Output Structure

Processed files create output directories with individual stem files:

```
/output/
  └── song_name/
      ├── vocals.wav
      ├── drums.wav
      ├── bass.wav
      ├── guitar.wav
      ├── piano.wav
      └── other.wav
```

## API Endpoints

### GET /

Serves the web UI interface for browser-based stem separation.

### POST /separate

Upload an audio file and receive a job ID for tracking progress (web UI) or a ZIP archive (legacy API).

**Request:**
- Content-Type: `multipart/form-data`
- Field name: `file`
- Accepted formats: MP3, WAV, FLAC, M4A
- Form fields:
  - `stems` (JSON array, optional): Selected stems to extract (e.g., `["vocals", "drums"]`)
  - `preset` (string, optional): Performance preset - fast, balanced, quality, or ultra (overrides individual settings)
  - `model` (string, optional): Model to use (default: from preset or "htdemucs")
  - `raw` (bool, optional): Skip cleanup (default: from preset or false)
  - `repair_glitch` (bool, optional): Enable glitch repair (default: from preset or false)
  - `repair_mode` (string, optional): Glitch repair mode - fast, standard, or thorough (default: standard)
  - `format` (string, default: "wav"): Output format (wav or flac)
  - `device` (string, optional): Device to use (cpu, cuda, or auto)

**Response (Web UI mode):**
- Content-Type: `application/json`
- JSON with `job_id` for tracking progress:
```json
{
  "job_id": "uuid-here",
  "status": "queued"
}
```

**Response (Legacy API mode):**
- Content-Type: `application/zip`
- ZIP file containing separated stems

### GET /progress/{job_id}

Server-Sent Events (SSE) endpoint for real-time job progress updates.

**Response:**
- Content-Type: `text/event-stream`
- SSE stream with progress updates:
```json
{
  "status": "processing",
  "message": "Separating audio stems...",
  "progress": 45,
  "log_messages": [
    {
      "timestamp": "2024-01-01T12:00:00",
      "message": "Job created for song.mp3",
      "status": "queued",
      "progress": 0
    },
    {
      "timestamp": "2024-01-01T12:00:05",
      "message": "Saving uploaded file...",
      "status": "processing",
      "progress": 5
    }
  ]
}
```

### GET /download/{job_id}

Download processed stems ZIP file for a completed job.

**Response:**
- Content-Type: `application/zip`
- ZIP file containing separated stems

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "device_info": {
    "cuda_available": true,
    "device_count": 1,
    "device_name": "NVIDIA GeForce RTX 3080",
    "gpu_memory_total_mb": 10240,
    "gpu_memory_allocated_mb": 512,
    "gpu_memory_cached_mb": 1024
  }
}
```

### GET /presets

Get available performance presets and their configurations.

**Response:**
```json
{
  "presets": {
    "fast": {
      "name": "fast",
      "model": "htdemucs",
      "raw": true,
      "repair_glitch": false,
      "repair_mode": "none",
      "description": "Fastest processing...",
      "estimated_time_3min_song": "5-15 seconds (GPU)"
    },
    ...
  },
  "default": "balanced"
}
```

## Volume Mounting

The container expects two volume mounts:

- `/input`: Directory containing input audio files
- `/output`: Directory where separated stems will be saved

Example:

```bash
docker run --rm \
  -v /path/to/your/input:/input \
  -v /path/to/your/output:/output \
  partitone "song.mp3"
```

## Performance Presets

PartiTone includes four performance presets optimized for different use cases:

### Fast Preset
- **Model**: htdemucs (faster model)
- **Cleanup**: Disabled (raw output)
- **Glitch Repair**: Disabled
- **Best for**: Quick previews, testing
- **Estimated time**: 5-15 seconds for 3-minute song (GPU)

### Balanced Preset (Default)
- **Model**: htdemucs
- **Cleanup**: Enabled (light cleanup)
- **Glitch Repair**: Disabled
- **Best for**: General use, good balance of quality and speed
- **Estimated time**: 15-30 seconds for 3-minute song (GPU)

### Quality Preset
- **Model**: htdemucs_ft (fine-tuned, higher quality)
- **Cleanup**: Enabled
- **Glitch Repair**: Enabled (standard mode)
- **Best for**: Production use, higher quality output
- **Estimated time**: 30-60 seconds for 3-minute song (GPU)

### Ultra Preset
- **Model**: htdemucs_ft (fine-tuned)
- **Cleanup**: Enabled
- **Glitch Repair**: Enabled (thorough mode)
- **Best for**: Maximum quality, professional use
- **Estimated time**: 60-90 seconds for 3-minute song (GPU)

**Note**: Times are estimates for GPU-accelerated processing. CPU processing will be significantly slower (2-5 minutes or more).

## Processing Stages

### Light Cleanup (Default)

By default, PartiTone applies light cleanup processing to preserve the human element while handling technical issues:

- **Silence Trimming**: Removes leading and trailing silence
- **DC Offset Removal**: Eliminates DC bias
- **Phase Alignment**: Aligns stems to reduce phase cancellation
- **Loudness Balancing**: Soft stem-level balancing (target: -23 LUFS)

**What it does NOT do:**
- No pitch correction
- No compression
- No EQ shaping
- No dynamics processing

Use `--raw` flag or Fast preset to skip cleanup and get raw Demucs output.

### Glitch Repair (Optional)

Enable with `--repair-glitch` flag or Quality/Ultra presets to reduce minor digital artifacts:

- **Click/Pop Repair**: Detects and repairs clicks and pops using interpolation
- **Stutter Repair**: Detects and repairs stutters and micro-glitches using crossfade

**Repair Modes:**
- **Fast**: Higher thresholds, fewer detections, faster processing (~5-10 seconds)
- **Standard**: Balanced detection and repair (~10-20 seconds)
- **Thorough**: Lower thresholds, more detections, comprehensive repair (~15-30 seconds)

Preserves timing, pitch, and dynamics while reducing technical artifacts.

## Performance Optimization

### GPU Acceleration

GPU acceleration is the most important factor for performance:
- **With GPU**: 3-minute song can process in 5-90 seconds (depending on preset)
- **Without GPU**: 3-minute song takes 2-5+ minutes

PartiTone automatically detects and uses GPU when available. Multi-GPU systems will use the first available GPU.

### Parallel Processing

PartiTone automatically parallelizes processing for improved performance:

#### Batch File Processing
- Multiple files are processed simultaneously using thread pools
- **GPU systems**: Up to 4 files processed in parallel (to avoid GPU memory issues)
- **CPU systems**: Uses all available CPU cores
- **Expected speedup**: 2-4x faster for batch processing (depending on CPU cores)

#### Stem Processing
- Cleanup operations (trim silence, DC offset removal) run in parallel across stems
- Glitch repair operations run in parallel across stems
- **Expected speedup**: 2-4x faster for cleanup/repair (depending on number of stems)
- Loudness balancing remains sequential (requires all stems together)

Parallel processing is automatically enabled and optimized based on:
- Number of files (batch mode)
- Number of stems
- Available CPU cores
- GPU availability

### Performance Monitoring

Performance metrics are tracked and displayed:
- Separation time (Demucs processing)
- Cleanup time (if enabled)
- Repair time (if enabled)
- Total processing time

Metrics are shown in:
- CLI output after processing
- Web UI processing log accordion
- Job completion messages

## Examples

See `examples/sample_cmds.md` for more detailed usage examples.

## System Requirements

- **CPU**: Any modern CPU (processing will be slower)
- **GPU**: NVIDIA GPU with CUDA 11.0+ for faster processing
- **RAM**: Minimum 4GB, recommended 8GB+
- **Disk**: ~500MB for Docker image, plus space for output files

## Platform Support

- Linux (Ubuntu, Debian, etc.)
- macOS (Intel and Apple Silicon)
- Windows (via WSL2)
- Synology NAS
- Unraid
- UGreen NAS

## License

MIT License

