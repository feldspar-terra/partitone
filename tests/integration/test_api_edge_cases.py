"""Integration tests for API edge cases and error handling."""

import pytest
import json
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

from app.api import app
from app.background import job_manager


class TestAPIEdgeCases:
    """Tests for API edge cases."""
    
    def test_separate_empty_stems_array(self, test_client, sample_audio_file):
        """Test /separate with empty stems array."""
        with open(sample_audio_file, "rb") as f:
            response = test_client.post(
                "/separate",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"stems": json.dumps([])}
            )
        
        assert response.status_code == 400
        assert "at least one valid stem" in response.json()["detail"].lower()
    
    def test_separate_invalid_stems_json(self, test_client, sample_audio_file):
        """Test /separate with invalid stems JSON."""
        with open(sample_audio_file, "rb") as f:
            response = test_client.post(
                "/separate",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"stems": "invalid json"}
            )
        
        assert response.status_code == 400
        assert "invalid stems parameter" in response.json()["detail"].lower()
    
    def test_separate_invalid_stem_names(self, test_client, sample_audio_file):
        """Test /separate with invalid stem names."""
        with open(sample_audio_file, "rb") as f:
            response = test_client.post(
                "/separate",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"stems": json.dumps(["invalid_stem", "another_invalid"])}
            )
        
        assert response.status_code == 400
        assert "at least one valid stem" in response.json()["detail"].lower()
    
    def test_separate_mixed_valid_invalid_stems(self, test_client, sample_audio_file):
        """Test /separate with mix of valid and invalid stems."""
        with open(sample_audio_file, "rb") as f:
            response = test_client.post(
                "/separate",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"stems": json.dumps(["vocals", "invalid_stem", "drums"])}
            )
        
        # Should succeed with only valid stems
        assert response.status_code == 200
        data = response.json()
        job = job_manager.get_job(data["job_id"])
        assert "vocals" in job.selected_stems
        assert "drums" in job.selected_stems
        assert "invalid_stem" not in job.selected_stems
    
    def test_separate_invalid_format(self, test_client, sample_audio_file):
        """Test /separate with invalid format parameter."""
        with open(sample_audio_file, "rb") as f:
            response = test_client.post(
                "/separate",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"format": "mp3"}  # Invalid format
            )
        
        # Should default to wav
        assert response.status_code == 200
        data = response.json()
        job = job_manager.get_job(data["job_id"])
        assert job.format == "wav"  # Should default to wav
    
    def test_progress_nonexistent_job(self, test_client):
        """Test /progress with nonexistent job ID."""
        response = test_client.get("/progress/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 200  # SSE returns 200
        
        # Check that error message is in the stream
        content = response.text
        assert "not found" in content.lower() or "failed" in content.lower()
    
    def test_download_nonexistent_job(self, test_client):
        """Test /download with nonexistent job ID."""
        response = test_client.get("/download/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404
    
    def test_separate_missing_file(self, test_client):
        """Test /separate without file."""
        response = test_client.post("/separate")
        assert response.status_code == 422  # Validation error
    
    def test_separate_with_all_stems(self, test_client, sample_audio_file):
        """Test /separate with all available stems."""
        all_stems = ["vocals", "drums", "bass", "guitar", "piano", "other"]
        
        with open(sample_audio_file, "rb") as f:
            response = test_client.post(
                "/separate",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"stems": json.dumps(all_stems)}
            )
        
        assert response.status_code == 200
        data = response.json()
        job = job_manager.get_job(data["job_id"])
        assert len(job.selected_stems) == 6
        assert set(job.selected_stems) == set(all_stems)
    
    def test_separate_boolean_strings(self, test_client, sample_audio_file):
        """Test /separate with boolean parameters as strings."""
        with open(sample_audio_file, "rb") as f:
            response = test_client.post(
                "/separate",
                files={"file": ("test.wav", f, "audio/wav")},
                data={
                    "raw": "true",
                    "repair_glitch": "false"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        job = job_manager.get_job(data["job_id"])
        assert job.raw is True
        assert job.repair_glitch is False

