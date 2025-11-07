#!/bin/bash
set -e

# PartiTone Docker Entrypoint Script
# Handles both CLI and API modes

if [ "$#" -eq 0 ]; then
    # No arguments, start API server
    echo "Starting PartiTone API server..."
    cd /app && python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
elif [ "$1" = "api" ]; then
    # Explicit API mode
    echo "Starting PartiTone API server..."
    cd /app && python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
else
    # Arguments provided, run CLI
    echo "Running PartiTone CLI..."
    cd /app && python -m app.cli "$@"
fi

