# PartiTone Usage Examples

## Basic CLI Usage

### Single File Processing

Process a single audio file:

```bash
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "song.mp3"
```

Process with custom output directory:

```bash
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/stems:/output" \
  partitone "/input/song.mp3" --output /output
```

### Custom Options

Specify output format (FLAC):

```bash
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "song.wav" --format flac
```

Specify number of stems:

```bash
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "song.mp3" --stems 6
```

Force CPU mode:

```bash
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "song.mp3" --device cpu
```

### Processing Options

Skip light cleanup (raw Demucs output):

```bash
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "song.mp3" --raw
```

Enable glitch repair:

```bash
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "song.mp3" --repair-glitch
```

Combine options (raw output with glitch repair):

```bash
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "song.mp3" --raw --repair-glitch
```

Default behavior (light cleanup applied automatically):

```bash
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "song.mp3"
```

### Batch Processing

Process all audio files in a directory:

```bash
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "/input" --batch
```

Or automatically detect directory:

```bash
docker run --rm \
  -v "$PWD/music:/input" \
  -v "$PWD/stems:/output" \
  partitone "/input" --batch
```

## API Usage

### Start API Server

Basic server (CPU):

```bash
docker run --rm \
  -p 8000:8000 \
  partitone api
```

With GPU support:

```bash
docker run --rm \
  -p 8000:8000 \
  --gpus all \
  partitone api
```

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
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

### Separate Audio File

Basic request:

```bash
curl -F "file=@song.mp3" \
  http://localhost:8000/separate \
  --output stems.zip
```

With custom model and format:

```bash
curl -F "file=@song.mp3" \
  -F "model=htdemucs" \
  -F "format=flac" \
  http://localhost:8000/separate \
  --output stems.zip
```

Skip cleanup (raw output):

```bash
curl -F "file=@song.mp3" \
  -F "raw=true" \
  http://localhost:8000/separate \
  --output stems.zip
```

Enable glitch repair:

```bash
curl -F "file=@song.mp3" \
  -F "repair_glitch=true" \
  http://localhost:8000/separate \
  --output stems.zip
```

Combine options:

```bash
curl -F "file=@song.mp3" \
  -F "raw=false" \
  -F "repair_glitch=true" \
  http://localhost:8000/separate \
  --output stems.zip
```

Force CPU mode:

```bash
curl -F "file=@song.mp3" \
  -F "device=cpu" \
  http://localhost:8000/separate \
  --output stems.zip
```

### Extract ZIP Archive

After downloading the ZIP:

```bash
unzip stems.zip -d output_directory/
```

## Volume Mounting Examples

### Linux/macOS

```bash
# Create input/output directories
mkdir -p ~/partitone/input ~/partitone/output

# Copy audio files
cp ~/Music/*.mp3 ~/partitone/input/

# Run processing
docker run --rm \
  -v ~/partitone/input:/input \
  -v ~/partitone/output:/output \
  partitone "song.mp3"
```

### Windows (WSL)

```bash
# In WSL
docker run --rm \
  -v /mnt/c/Users/YourName/Music:/input \
  -v /mnt/c/Users/YourName/Stems:/output \
  partitone "song.mp3"
```

### Synology NAS

Using Docker GUI:
1. Create volumes mapping `/input` and `/output` to NAS directories
2. Run container with command: `song.mp3`

Using SSH:

```bash
docker run --rm \
  -v /volume1/music:/input \
  -v /volume1/stems:/output \
  partitone "song.mp3"
```

### Unraid

```bash
docker run --rm \
  -v /mnt/user/music:/input \
  -v /mnt/user/stems:/output \
  partitone "song.mp3"
```

## Advanced Examples

### Process Multiple Formats

```bash
# Process MP3
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "song.mp3" --format wav

# Process FLAC
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "song.flac" --format flac
```

### Monitor Progress

```bash
docker run --rm \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "song.mp3" 2>&1 | tee processing.log
```

### Background Processing

```bash
docker run -d \
  --name partitone-job \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  partitone "song.mp3"

# Check logs
docker logs partitone-job

# Clean up
docker rm partitone-job
```

## Troubleshooting

### Check GPU Availability

```bash
docker run --rm --gpus all partitone python -c "import torch; print(torch.cuda.is_available())"
```

### Test API Endpoint

```bash
# Health check
curl http://localhost:8000/health

# Root endpoint
curl http://localhost:8000/
```

### View Container Logs

```bash
# For API server
docker logs <container_id>

# For CLI jobs
docker logs <container_id>
```

