"""
test_phase1_upload.py — Comprehensive acceptance tests for Phase 1 (Foundation + Upload).

Acceptance Matrix (13 Required Tests):
Row 1:  Valid MP3 upload, under caps -> 201/200, metadata saved
Row 2:  Valid WAV upload, under caps -> 201/200, metadata saved
Row 3:  Valid M4A upload, under caps -> 201/200, metadata saved
Row 4:  Valid MP4 video upload, under video cap -> 201/200, metadata saved
Row 5:  Audio file > 50 MB -> HTTP 413, structured JSON error
Row 6:  Video file > 400 MB -> HTTP 413, structured JSON error
Row 7:  Unsupported extension (.txt) -> HTTP 415, structured JSON error
Row 8:  Unsupported extension (.exe) -> HTTP 415, structured JSON error
Row 9:  Unsupported extension (.pdf) -> HTTP 415, structured JSON error
Row 10: Corrupt/malformed audio file -> Rejected, structured JSON error, no crash
Row 11: Recording duration > 30 min -> Rejected with duration-violation detail
Row 12: GET /api/recordings/{id} for valid upload -> Returns full metadata
Row 13: GET /api/recordings/{id} for nonexistent ID -> 404, structured JSON error
"""

import os
import tempfile
import subprocess
import pytest
from fastapi.testclient import TestClient
import imageio_ffmpeg

from app.main import app
from app.config import (
    MAX_AUDIO_SIZE_BYTES,
    MAX_VIDEO_SIZE_BYTES,
    MAX_DURATION_SECONDS,
)


@pytest.fixture(scope="session")
def ffmpeg_bin():
    return imageio_ffmpeg.get_ffmpeg_exe()


def generate_media_bytes(ffmpeg_bin: str, extension: str, duration_sec: float = 1.0) -> bytes:
    """Generate real, valid media file bytes using FFmpeg for authentic format testing."""
    tmp_path = tempfile.mktemp(suffix=extension)
    try:
        if extension == ".wav":
            cmd = [ffmpeg_bin, "-y", "-f", "lavfi", "-i", f"sine=frequency=1000:duration={duration_sec}", tmp_path]
        elif extension == ".mp3":
            cmd = [ffmpeg_bin, "-y", "-f", "lavfi", "-i", f"sine=frequency=1000:duration={duration_sec}", tmp_path]
        elif extension == ".m4a":
            cmd = [ffmpeg_bin, "-y", "-f", "lavfi", "-i", f"sine=frequency=1000:duration={duration_sec}", "-c:a", "aac", tmp_path]
        elif extension == ".mp4":
            cmd = [
                ffmpeg_bin, "-y",
                "-f", "lavfi", "-i", f"testsrc=duration={duration_sec}:size=160x120:rate=10",
                "-f", "lavfi", "-i", f"sine=frequency=1000:duration={duration_sec}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                tmp_path
            ]
        else:
            raise ValueError(f"Unsupported test generator format: {extension}")

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


# ============================================================================
# Row 1: Valid MP3 upload, under caps
# ============================================================================
def test_row1_valid_mp3_upload(client, ffmpeg_bin):
    mp3_bytes = generate_media_bytes(ffmpeg_bin, ".mp3", duration_sec=2.0)
    filename = "lecture_sample.mp3"

    response = client.post("/api/upload", files={"file": (filename, mp3_bytes, "audio/mpeg")})
    assert response.status_code in (200, 201), response.text

    data = response.json()
    assert data["status"] == "success"
    rec = data["data"]
    assert rec["filename"] == filename
    assert rec["file_format"] == ".mp3"
    assert rec["media_type"] == "audio"
    assert rec["file_size_bytes"] == len(mp3_bytes)
    assert rec["duration_seconds"] > 0
    assert rec["status"] == "UPLOADED"
    assert "id" in rec
    assert "created_at" in rec


# ============================================================================
# Row 2: Valid WAV upload, under caps
# ============================================================================
def test_row2_valid_wav_upload(client, ffmpeg_bin):
    wav_bytes = generate_media_bytes(ffmpeg_bin, ".wav", duration_sec=3.0)
    filename = "meeting_audio.wav"

    response = client.post("/api/upload", files={"file": (filename, wav_bytes, "audio/wav")})
    assert response.status_code in (200, 201), response.text

    data = response.json()
    assert data["status"] == "success"
    rec = data["data"]
    assert rec["filename"] == filename
    assert rec["file_format"] == ".wav"
    assert rec["media_type"] == "audio"
    assert rec["file_size_bytes"] == len(wav_bytes)
    assert abs(rec["duration_seconds"] - 3.0) < 0.2
    assert rec["status"] == "UPLOADED"


# ============================================================================
# Row 3: Valid M4A upload, under caps
# ============================================================================
def test_row3_valid_m4a_upload(client, ffmpeg_bin):
    m4a_bytes = generate_media_bytes(ffmpeg_bin, ".m4a", duration_sec=2.5)
    filename = "voice_memo.m4a"

    response = client.post("/api/upload", files={"file": (filename, m4a_bytes, "audio/mp4")})
    assert response.status_code in (200, 201), response.text

    data = response.json()
    assert data["status"] == "success"
    rec = data["data"]
    assert rec["filename"] == filename
    assert rec["file_format"] == ".m4a"
    assert rec["media_type"] == "audio"
    assert rec["file_size_bytes"] == len(m4a_bytes)
    assert rec["duration_seconds"] > 0
    assert rec["status"] == "UPLOADED"


# ============================================================================
# Row 4: Valid MP4 video upload, under video cap
# ============================================================================
def test_row4_valid_mp4_video_upload(client, ffmpeg_bin):
    mp4_bytes = generate_media_bytes(ffmpeg_bin, ".mp4", duration_sec=2.0)
    filename = "class_recording.mp4"

    response = client.post("/api/upload", files={"file": (filename, mp4_bytes, "video/mp4")})
    assert response.status_code in (200, 201), response.text

    data = response.json()
    assert data["status"] == "success"
    rec = data["data"]
    assert rec["filename"] == filename
    assert rec["file_format"] == ".mp4"
    assert rec["media_type"] == "video"
    assert rec["file_size_bytes"] == len(mp4_bytes)
    assert rec["duration_seconds"] > 0
    assert rec["status"] == "UPLOADED"


# ============================================================================
# Row 5: Audio file > 50 MB
# ============================================================================
def test_row5_audio_file_oversized_50mb(client, monkeypatch, ffmpeg_bin):
    from app.services import audio_validator
    test_max = 5000  # 5 KB for test
    monkeypatch.setattr(audio_validator, "MAX_AUDIO_SIZE_BYTES", test_max)

    wav_bytes = generate_media_bytes(ffmpeg_bin, ".wav", duration_sec=2.0)  # > 50 KB
    assert len(wav_bytes) > test_max

    response = client.post("/api/upload", files={"file": ("oversized_audio.wav", wav_bytes, "audio/wav")})
    assert response.status_code == 413

    data = response.json()
    assert data["status"] == "error"
    assert data["code"] == "FILE_TOO_LARGE"
    assert "exceeds maximum allowed limit" in data["message"]
    assert "size_bytes" in data["details"]


# ============================================================================
# Row 6: Video file > 400 MB
# ============================================================================
def test_row6_video_file_oversized_400mb(client, monkeypatch, ffmpeg_bin):
    from app.services import audio_validator
    test_max = 10000  # 10 KB for test
    monkeypatch.setattr(audio_validator, "MAX_VIDEO_SIZE_BYTES", test_max)

    mp4_bytes = generate_media_bytes(ffmpeg_bin, ".mp4", duration_sec=2.0)  # > 15 KB
    assert len(mp4_bytes) > test_max

    response = client.post("/api/upload", files={"file": ("oversized_video.mp4", mp4_bytes, "video/mp4")})
    assert response.status_code == 413

    data = response.json()
    assert data["status"] == "error"
    assert data["code"] == "FILE_TOO_LARGE"
    assert "exceeds maximum allowed limit" in data["message"]


# ============================================================================
# Row 7: Unsupported extension (.txt)
# ============================================================================
def test_row7_unsupported_txt(client):
    response = client.post("/api/upload", files={"file": ("transcript.txt", b"plain text", "text/plain")})
    assert response.status_code == 415

    data = response.json()
    assert data["status"] == "error"
    assert data["code"] == "UNSUPPORTED_FORMAT"
    assert "Unsupported file format" in data["message"]


# ============================================================================
# Row 8: Unsupported extension (.exe)
# ============================================================================
def test_row8_unsupported_exe(client):
    response = client.post("/api/upload", files={"file": ("installer.exe", b"MZbinarycode", "application/x-msdownload")})
    assert response.status_code == 415

    data = response.json()
    assert data["status"] == "error"
    assert data["code"] == "UNSUPPORTED_FORMAT"


# ============================================================================
# Row 9: Unsupported extension (.pdf)
# ============================================================================
def test_row9_unsupported_pdf(client):
    response = client.post("/api/upload", files={"file": ("slides.pdf", b"%PDF-1.4 sample", "application/pdf")})
    assert response.status_code == 415

    data = response.json()
    assert data["status"] == "error"
    assert data["code"] == "UNSUPPORTED_FORMAT"


# ============================================================================
# Row 10: Corrupt/malformed audio file (valid extension, garbage bytes)
# ============================================================================
def test_row10_corrupt_audio_rejected(client):
    corrupt_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00garbage_corrupt_data_here"

    response = client.post("/api/upload", files={"file": ("corrupt_audio.wav", corrupt_bytes, "audio/wav")})
    assert response.status_code in (400, 422)

    data = response.json()
    assert data["status"] == "error"
    assert data["code"] == "CORRUPT_FILE"
    assert "validation failed" in data["message"].lower()


# ============================================================================
# Row 11: Recording duration > 30 min
# ============================================================================
def test_row11_duration_exceeded_30min(client, monkeypatch, ffmpeg_bin):
    from app.services import audio_validator
    test_max_duration = 2.0  # 2 seconds for test
    monkeypatch.setattr(audio_validator, "MAX_DURATION_SECONDS", test_max_duration)

    wav_bytes = generate_media_bytes(ffmpeg_bin, ".wav", duration_sec=5.0)

    response = client.post("/api/upload", files={"file": ("long_speech.wav", wav_bytes, "audio/wav")})
    assert response.status_code == 400

    data = response.json()
    assert data["status"] == "error"
    assert data["code"] == "DURATION_EXCEEDED"
    assert "exceeds maximum allowed limit" in data["message"]
    assert "duration_seconds" in data["details"]


# ============================================================================
# Row 12: GET /api/recordings/{id} for a valid upload
# ============================================================================
def test_row12_get_recording_valid_id(client, ffmpeg_bin):
    wav_bytes = generate_media_bytes(ffmpeg_bin, ".wav", duration_sec=1.5)
    up_res = client.post("/api/upload", files={"file": ("get_test.wav", wav_bytes, "audio/wav")})
    assert up_res.status_code in (200, 201)
    rec_id = up_res.json()["data"]["id"]

    get_res = client.get(f"/api/recordings/{rec_id}")
    assert get_res.status_code == 200

    data = get_res.json()
    assert data["id"] == rec_id
    assert data["filename"] == "get_test.wav"
    assert data["file_format"] == ".wav"
    assert data["media_type"] == "audio"
    assert data["status"] == "UPLOADED"


# ============================================================================
# Row 13: GET /api/recordings/{id} for a nonexistent ID
# ============================================================================
def test_row13_get_recording_nonexistent_id(client):
    nonexistent_id = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"/api/recordings/{nonexistent_id}")
    assert response.status_code == 404

    data = response.json()
    assert data["status"] == "error"
    assert "not found" in data["message"].lower()
