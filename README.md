# PartiTone

A Dockerized audio stem separation tool that extracts individual stems (vocals, drums, bass, guitar, piano, other) from audio files using Demucs v4. Includes default light cleanup processing to preserve the human element while handling technical issues.

## Features

- **CLI Tool**: Process audio files from the command line
- **REST API**: Separate stems via HTTP API with file upload
- **Web UI**: Modern, user-friendly web interface with real-time progress tracking
- **Processing Log Accordion**: Expandable log viewer showing detailed processing updates with timestamps
- **GPU Acceleration**: Automatic CUDA detection and GPU acceleration when available
- **Batch Processing**: Process entire folders of audio files
- **Multiple Formats**: Supports MP3, WAV, FLAC, and M4A input/output
- **Light Cleanup** (default): Silence trimming, DC offset removal, phase alignment, soft loudness balancing
- **Glitch Repair** (optional): Reduce clicks, pops, stutters, and micro-glitches
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

Process with custom options:

```bash
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "song.mp3" --stems 6 --format wav --output /output
```

Process a batch folder:

```bash
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "/input" --batch
```

#### CLI Options

- `--stems N`: Number of stems to extract (default: 6)
- `--model MODEL`: Model to use (default: "htdemucs")
- `--output DIR`: Output directory (default: "/output")
- `--format wav|flac`: Output format (default: "wav")
- `--batch`: Process folder instead of single file
- `--raw`: Skip light cleanup entirely (output raw Demucs stems)
- `--repair-glitch`: Enable glitch repair stage (reduce clicks, pops, stutters)

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
- Real-time progress bar
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
  - `raw` (bool, default: false): Skip cleanup
  - `repair_glitch` (bool, default: false): Enable glitch repair
  - `format` (string, default: "wav"): Output format (wav or flac)
  - `model` (string, default: "htdemucs"): Model to use
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
    "device_name": "NVIDIA GeForce RTX 3080"
  }
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

Use `--raw` flag to skip cleanup and get raw Demucs output.

### Glitch Repair (Optional)

Enable with `--repair-glitch` flag to reduce minor digital artifacts:

- **Click/Pop Repair**: Detects and repairs clicks and pops using interpolation
- **Stutter Repair**: Detects and repairs stutters and micro-glitches using crossfade

Preserves timing, pitch, and dynamics while reducing technical artifacts.

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

