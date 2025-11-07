FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy tests directory
COPY tests/ ./tests/

# Ensure static directory exists (even if empty)
RUN mkdir -p app/static

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set Python path
ENV PYTHONPATH=/app

ENTRYPOINT ["/entrypoint.sh"]

# Expose API port
EXPOSE 8000

