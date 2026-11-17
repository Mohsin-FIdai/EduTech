"""
test_phase3_chunking.py — Acceptance tests for Phase 3 (Intelligent Chunking).

Acceptance Matrix:
1. Gated Silence-Aware Boundary Splitting:
   - Chunk boundaries align strictly with silence pauses (RMS <= -30.0 dBFS) and never bisect speech bursts.
2. Hard <= 30.0s Whisper Ceiling on Continuous Monologue:
   - Continuous 60s speech with no silence splits at RMS valleys, strictly <= 30.0s.
3. Short Recording Single-Chunk Handling:
   - 14.0s recording (< 20.0s target) produces exactly 1 clean chunk without artificial silence padding.
4. Trailing Audio (< 2.0s) Merge:
   - Trailing 0.8s tail merges cleanly into previous chunk; zero audio dropped.
5. Zero-Loss Full Waveform Coverage:
   - Union of [start_time, end_time] across all chunks exactly equals [0.0, Total Duration] with 0 gaps.
6. Audio Standard & Overlap:
   - All exported chunk WAVs are strictly 16,000 Hz, 1 channel (mono), 16-bit PCM.
7. REST API Endpoints:
   - POST /api/recordings/{id}/chunk, GET /api/recordings/{id}/chunks, GET chunk audio stream.
8. 10+ Minute Recording 5-Point Boundary Spot-Check:
   - Probes 5 consecutive split points on a 600s (10 min) lecture track; confirms RMS <= -30.0 dBFS at every boundary.
"""

import os
import tempfile
from pathlib import Path
import subprocess
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
import imageio_ffmpeg

from app.main import app
from app.services.audio_chunker import AudioChunker
from app.services.audio_preprocessor import AudioPreprocessor
from app.models import Recording, Chunk, RecordingStatus
from app.database import get_db


@pytest.fixture(scope="session")
def ffmpeg_bin():
    return imageio_ffmpeg.get_ffmpeg_exe()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def create_preprocessed_wav(audio_data: np.ndarray, sample_rate: int = 16000) -> str:
    """Helper to write 16kHz float32 audio to a temp WAV file."""
    tmp = tempfile.mktemp(suffix="_preprocessed.wav")
    sf.write(tmp, audio_data, sample_rate, subtype="PCM_16")
    return tmp


# ============================================================================
# Test 1: Silence-Aware Boundary Splitting
# ============================================================================
def test_silence_aware_boundary_splitting(client):
    """
    Generate 50s audio:
    - 0.0s - 23.0s: Speech (440Hz tone)
    - 23.0s - 25.0s: Natural Silence Gap (2.0s silence pause)
    - 25.0s - 50.0s: Speech (880Hz tone)
    Verify split occurs strictly within the silence gap (23.0s - 25.0s) with RMS <= -30 dBFS.
    """
    sr = 16000
    duration = 50.0
    t = np.linspace(0, duration, int(sr * duration), False)

    audio = np.zeros_like(t)
    audio[0:int(sr * 23.0)] = 0.5 * np.sin(2 * np.pi * 440 * t[0:int(sr * 23.0)])
    # 23.0 - 25.0 remains exact 0.0 (silence)
    audio[int(sr * 25.0):] = 0.5 * np.sin(2 * np.pi * 880 * t[int(sr * 25.0):])

    wav_path = create_preprocessed_wav(audio, sr)
    try:
        with open(wav_path, "rb") as f:
            up_res = client.post("/api/upload", files={"file": ("silence_lecture.wav", f.read(), "audio/wav")})
        rec_id = up_res.json()["data"]["id"]

        # Run chunking
        chunk_res = client.post(f"/api/recordings/{rec_id}/chunk")
        assert chunk_res.status_code == 200
        data = chunk_res.json()

        assert data["total_chunks"] == 2
        chunks = data["chunks"]

        # First chunk end and second chunk start must be in silence gap [23.0, 25.0]
        split_point = chunks[0]["end_time"]
        assert 23.0 <= split_point <= 25.0, f"Split point {split_point}s not in silence gap [23.0, 25.0]"
        assert chunks[1]["start_time"] == split_point

        # Probe RMS energy in audio at split point
        split_sample = int(round(split_point * sr))
        local_window = audio[max(0, split_sample - 800): min(len(audio), split_sample + 800)]
        local_rms_dbfs = 20 * np.log10(np.sqrt(np.mean(local_window ** 2)) + 1e-12)
        assert local_rms_dbfs <= -30.0, f"RMS at split point {local_rms_dbfs} dBFS exceeds -30 dBFS"
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ============================================================================
# Test 2: Hard <= 30.0s Ceiling on Continuous Monologue (Priority 2 Fallback)
# ============================================================================
def test_hard_30s_ceiling_continuous_monologue(client):
    """
    Generate 60.0s of continuous tone (no silence gaps at all).
    Verify all chunks are strictly <= 30.0s (Whisper ceiling) and use RMS valley fallback.
    """
    sr = 16000
    duration = 60.0
    t = np.linspace(0, duration, int(sr * duration), False)

    # Continuous tone with an amplitude dip at 24.5s
    audio = 0.5 * np.sin(2 * np.pi * 500 * t)
    # Create dip at 24.0s - 25.0s
    dip_idx = (t >= 24.0) & (t <= 25.0)
    audio[dip_idx] = audio[dip_idx] * 0.1

    wav_path = create_preprocessed_wav(audio, sr)
    try:
        with open(wav_path, "rb") as f:
            up_res = client.post("/api/upload", files={"file": ("continuous_speech.wav", f.read(), "audio/wav")})
        rec_id = up_res.json()["data"]["id"]

        chunk_res = client.post(f"/api/recordings/{rec_id}/chunk")
        assert chunk_res.status_code == 200
        data = chunk_res.json()

        for ch in data["chunks"]:
            assert ch["duration"] <= 30.0, f"Chunk duration {ch['duration']}s exceeds 30.0s ceiling"
            assert ch["duration"] >= 2.0

        # Verify split was made in the dip window
        split_point = data["chunks"][0]["end_time"]
        assert 23.5 <= split_point <= 25.5
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ============================================================================
# Test 3: Short Recording Single-Chunk Handling (< 20.0s)
# ============================================================================
def test_short_recording_single_chunk(client):
    """
    A 14.0s recording under target chunk duration (25.0s) must produce exactly 1 chunk
    spanning [0.0, 14.0s] without artificial padding or unnecessary splitting.
    """
    sr = 16000
    duration = 14.0
    t = np.linspace(0, duration, int(sr * duration), False)
    audio = 0.4 * np.sin(2 * np.pi * 440 * t)

    wav_path = create_preprocessed_wav(audio, sr)
    try:
        with open(wav_path, "rb") as f:
            up_res = client.post("/api/upload", files={"file": ("short_memo.wav", f.read(), "audio/wav")})
        rec_id = up_res.json()["data"]["id"]

        chunk_res = client.post(f"/api/recordings/{rec_id}/chunk")
        assert chunk_res.status_code == 200
        data = chunk_res.json()

        assert data["total_chunks"] == 1
        ch = data["chunks"][0]
        assert ch["chunk_index"] == 0
        assert ch["start_time"] == 0.0
        assert abs(ch["end_time"] - 14.0) <= 0.05
        assert abs(ch["duration"] - 14.0) <= 0.05
        assert ch["overlap_start"] == 0.0
        assert ch["overlap_end"] == 0.0
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ============================================================================
# Test 4: Trailing Audio (< 2.0s) Merge
# ============================================================================
def test_trailing_audio_merge_into_previous_chunk(client):
    """
    Generate 25.8s recording where splitting at 25.0s would leave a 0.8s trailing fragment (< 2.0s min).
    Verify the 0.8s tail is merged into the chunk, resulting in 1 chunk of 25.8s (<= 30.0s ceiling).
    """
    sr = 16000
    duration = 25.8
    t = np.linspace(0, duration, int(sr * duration), False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    wav_path = create_preprocessed_wav(audio, sr)
    try:
        with open(wav_path, "rb") as f:
            up_res = client.post("/api/upload", files={"file": ("trailing_audio.wav", f.read(), "audio/wav")})
        rec_id = up_res.json()["data"]["id"]

        chunk_res = client.post(f"/api/recordings/{rec_id}/chunk")
        assert chunk_res.status_code == 200
        data = chunk_res.json()

        # Should merge into a single 25.8s chunk because 25.8s <= 30.0s max limit
        assert data["total_chunks"] == 1
        ch = data["chunks"][0]
        assert ch["start_time"] == 0.0
        assert abs(ch["end_time"] - 25.8) <= 0.05
        assert abs(ch["duration"] - 25.8) <= 0.05
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ============================================================================
# Test 5: Zero Loss Full Waveform Coverage & Overlap
# ============================================================================
def test_zero_loss_full_waveform_coverage(client):
    """
    Generate 75.0s audio track.
    Verify:
    1. chunk[0].start_time == 0.0
    2. chunk[i].end_time == chunk[i+1].start_time (zero gap)
    3. chunk[-1].end_time == 75.0
    4. Sliced audio includes context overlap at boundaries.
    """
    sr = 16000
    duration = 75.0
    t = np.linspace(0, duration, int(sr * duration), False)
    audio = 0.4 * np.sin(2 * np.pi * 600 * t)

    # Place silence gaps every 24s
    for gap_start in [24.0, 48.0]:
        audio[int(sr * gap_start): int(sr * (gap_start + 1.0))] = 0.0

    wav_path = create_preprocessed_wav(audio, sr)
    try:
        with open(wav_path, "rb") as f:
            up_res = client.post("/api/upload", files={"file": ("coverage_75s.wav", f.read(), "audio/wav")})
        rec_id = up_res.json()["data"]["id"]

        chunk_res = client.post(f"/api/recordings/{rec_id}/chunk")
        assert chunk_res.status_code == 200
        data = chunk_res.json()
        chunks = data["chunks"]

        assert len(chunks) >= 3

        # 1. Start from 0.0
        assert chunks[0]["start_time"] == 0.0

        # 2. Continuous zero-gap coverage
        for i in range(len(chunks) - 1):
            assert chunks[i]["end_time"] == chunks[i + 1]["start_time"], f"Gap detected between chunk {i} and {i+1}"

        # 3. End matches total duration
        assert abs(chunks[-1]["end_time"] - 75.0) <= 0.05

        # 4. Context overlap checks
        assert chunks[0]["overlap_start"] == 0.0
        assert chunks[0]["overlap_end"] > 0.0
        assert chunks[1]["overlap_start"] > 0.0
        assert chunks[-1]["overlap_end"] == 0.0
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ============================================================================
# Test 6: Chunk Audio Format Standard & File Persistence
# ============================================================================
def test_chunk_audio_format_standard(client):
    """
    Verify all generated chunk WAV files on disk:
    1. Sample rate is exactly 16,000 Hz.
    2. Channels = 1 (Mono).
    3. File format is 16-bit PCM WAV.
    """
    sr = 16000
    duration = 45.0
    t = np.linspace(0, duration, int(sr * duration), False)
    audio = 0.5 * np.sin(2 * np.pi * 700 * t)
    audio[int(sr * 22.0): int(sr * 23.0)] = 0.0

    wav_path = create_preprocessed_wav(audio, sr)
    try:
        with open(wav_path, "rb") as f:
            up_res = client.post("/api/upload", files={"file": ("format_check.wav", f.read(), "audio/wav")})
        rec_id = up_res.json()["data"]["id"]

        chunk_res = client.post(f"/api/recordings/{rec_id}/chunk")
        assert chunk_res.status_code == 200
        chunks = chunk_res.json()["chunks"]

        for ch in chunks:
            fpath = Path(ch["file_path"])
            assert fpath.exists(), f"Chunk file {fpath} does not exist on disk"
            assert fpath.suffix == ".wav"

            data, ch_sr = sf.read(str(fpath))
            assert ch_sr == 16000, f"Chunk sample rate {ch_sr} != 16000"
            assert data.ndim == 1, "Chunk must be mono (1 channel)"
            info = sf.info(str(fpath))
            assert info.subtype == "PCM_16"
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ============================================================================
# Test 7: REST API Endpoints & Streaming Audio
# ============================================================================
def test_chunking_rest_endpoints(client):
    """
    Test POST /api/recordings/{id}/chunk, GET /api/recordings/{id}/chunks,
    and GET /api/recordings/{id}/chunks/{chunk_index}/audio.
    """
    sr = 16000
    duration = 35.0
    t = np.linspace(0, duration, int(sr * duration), False)
    audio = 0.4 * np.sin(2 * np.pi * 500 * t)
    audio[int(sr * 20.0): int(sr * 21.0)] = 0.0

    wav_path = create_preprocessed_wav(audio, sr)
    try:
        with open(wav_path, "rb") as f:
            up_res = client.post("/api/upload", files={"file": ("api_test.wav", f.read(), "audio/wav")})
        rec_id = up_res.json()["data"]["id"]

        # POST chunk
        post_res = client.post(f"/api/recordings/{rec_id}/chunk")
        assert post_res.status_code == 200

        # GET chunks list
        get_res = client.get(f"/api/recordings/{rec_id}/chunks")
        assert get_res.status_code == 200
        list_data = get_res.json()
        assert list_data["status"] == "success"
        assert list_data["total_chunks"] == len(list_data["data"])
        assert list_data["total_chunks"] >= 2

        # GET audio stream for chunk 0
        audio_stream_res = client.get(f"/api/recordings/{rec_id}/chunks/0/audio")
        assert audio_stream_res.status_code == 200
        assert audio_stream_res.headers["content-type"] == "audio/wav"
        assert len(audio_stream_res.content) > 1000

        # Check recording status in DB
        rec_check = client.get(f"/api/recordings/{rec_id}").json()
        assert rec_check["status"] == "CHUNKED"
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ============================================================================
# Test 8: 10+ Minute Recording 5-Point Boundary Spot-Check
# ============================================================================
def test_10min_lecture_5_boundary_spot_check(client):
    """
    Spot-check 5 consecutive chunk boundaries on a 10+ minute (600s) realistic lecture track:
    - Non-uniform, organic sentence durations and variable pause lengths.
    - Ambient room tone / mic noise floor (~ -48 dBFS) rather than synthetic 0-amplitude silence.
    - Multi-harmonic speech simulation with syllable modulation envelopes.
    - Measures exact RMS energy in +/- 100ms window around each of 5 consecutive split points.
    - Confirms RMS <= -30.0 dBFS (strictly avoiding cutting words in half).
    """
    sr = 16000
    duration = 600.0  # 10 minutes
    t = np.linspace(0, duration, int(sr * duration), False)

    # 1. Add realistic ambient room noise floor (~ -48 dBFS)
    np.random.seed(1337)
    room_tone = 0.0035 * np.random.normal(0, 1, len(t))
    audio = room_tone.copy()

    # 2. Organic non-uniform phrase structure with realistic variable pauses
    phrases = [
        (0.0, 21.8, 480),     # 21.8s speech
        (23.2, 46.4, 520),    # 1.4s pause -> 23.2s speech
        (47.5, 73.1, 450),    # 1.1s pause -> 25.6s speech
        (74.0, 97.6, 560),    # 0.9s pause -> 23.6s speech
        (99.2, 122.8, 510),   # 1.6s pause -> 23.6s speech
        (123.6, 149.1, 490),  # 0.8s pause -> 25.5s speech
        (150.8, 174.3, 530),  # 1.7s pause -> 23.5s speech
        (175.2, 198.9, 470),  # 0.9s pause -> 23.7s speech
    ]

    cur_t = 200.0
    while cur_t < 580.0:
        p_len = float(np.random.uniform(21.0, 26.0))
        pause_len = float(np.random.uniform(0.7, 1.8))
        phrases.append((cur_t, min(598.0, cur_t + p_len), int(np.random.uniform(400, 600))))
        cur_t += p_len + pause_len

    # Synthesize speech with syllable modulation envelopes
    for s_t, e_t, freq in phrases:
        if s_t < duration:
            e_t = min(e_t, duration)
            idx = slice(int(sr * s_t), int(sr * e_t))
            seg_t = t[idx]
            syllable_mod = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * seg_t)
            harmonics = 0.4 * np.sin(2 * np.pi * freq * seg_t) + 0.2 * np.sin(2 * np.pi * (freq * 2) * seg_t)
            audio[idx] += (harmonics * syllable_mod).astype(np.float32)

    wav_path = create_preprocessed_wav(audio, sr)
    try:
        with open(wav_path, "rb") as f:
            up_res = client.post("/api/upload", files={"file": ("realistic_lecture_10min.wav", f.read(), "audio/wav")})
        rec_id = up_res.json()["data"]["id"]

        # Preprocess and chunk
        client.post(f"/api/recordings/{rec_id}/preprocess")
        chunk_res = client.post(f"/api/recordings/{rec_id}/chunk")
        assert chunk_res.status_code == 200
        chunks = chunk_res.json()["chunks"]

        assert len(chunks) >= 20, f"Expected >= 20 chunks for 10-minute audio, got {len(chunks)}"

        print("\n--- 5-POINT BOUNDARY SPOT-CHECK (10-MINUTE REALISTIC RECORDING) ---")
        spot_check_indices = [1, 2, 3, 4, 5]
        for idx in spot_check_indices:
            ch_prev = chunks[idx - 1]
            ch_curr = chunks[idx]
            split_time = ch_prev["end_time"]

            # Measure RMS in +/- 100ms window around the cut
            sample_idx = int(round(split_time * sr))
            w_start = max(0, sample_idx - int(sr * 0.10))
            w_end = min(len(audio), sample_idx + int(sr * 0.10))
            cut_slice = audio[w_start:w_end]

            rms_val = float(np.sqrt(np.mean(cut_slice ** 2)))
            dbfs_val = float(max(-100.0, 20.0 * np.log10(rms_val + 1e-12)))

            p_s = ch_prev["start_time"]
            p_e = ch_prev["end_time"]
            p_d = ch_prev["duration"]
            c_s = ch_curr["start_time"]
            c_e = ch_curr["end_time"]
            c_d = ch_curr["duration"]

            print(f"  Boundary {idx}: Split Time = {split_time:6.2f}s | Window RMS = {dbfs_val:6.2f} dBFS | "
                  f"Chunk {idx-1} [{p_s:6.2f}s - {p_e:6.2f}s] ({p_d:5.2f}s) -> "
                  f"Chunk {idx} [{c_s:6.2f}s - {c_e:6.2f}s] ({c_d:5.2f}s)")

            # Strict Assertion: Every split point must be in natural pause (<= -30.0 dBFS)
            assert dbfs_val <= -30.0, f"Boundary {idx} at {split_time}s had RMS {dbfs_val} dBFS > -30 dBFS (cut speech)"
            assert ch_curr["duration"] <= 30.0
            assert ch_curr["duration"] >= 2.0

        print("--- ALL 5 SPOT-CHECK BOUNDARIES VERIFIED IN REALISTIC PAUSES (RMS <= -30 dBFS) ---\n")
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
