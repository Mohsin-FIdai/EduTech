"""
Audio Chunker Service — Intelligent, silence-aware audio partitioning.

Key Capabilities:
1. Target chunk sizing: 20.0s to 28.0s (strictly <= 30.0s for Whisper 30s context window).
2. Two-tier boundary selection:
   - Priority 1: Gated natural silence pauses (RMS <= -30.0 dBFS, duration >= 0.20s).
   - Priority 2: RMS energy valley fallback for continuous monologues (finds lowest energy frame).
3. Trailing audio merge (< 2.0s merged into previous chunk if <= 30.0s).
4. Short recording single-chunk handling (< 20.0s creates exactly 1 clean chunk).
5. Continuous zero-loss waveform coverage [0.0, Total Duration].
6. Context overlap tracking (overlap_start, overlap_end).
7. Database persistence and status transition to CHUNKED.
"""

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import soundfile as sf
from sqlalchemy.orm import Session

from app.config import (
    CHUNKS_DIR,
    TARGET_CHUNK_DURATION_SECONDS,
    MAX_CHUNK_DURATION_SECONDS,
    MIN_CHUNK_DURATION_SECONDS,
    DEFAULT_CHUNK_OVERLAP_SECONDS,
    SILENCE_THRESHOLD_DBFS,
    MIN_PAUSE_DURATION_SECONDS
)
from app.models import Recording, Chunk, RecordingStatus
from app.services.audio_preprocessor import AudioPreprocessor


class AudioChunkingException(Exception):
    """Exception raised for errors in audio chunking pipeline."""
    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


@dataclass
class ChunkPlanItem:
    chunk_index: int
    start_time: float
    end_time: float
    duration: float
    overlap_start: float
    overlap_end: float
    slice_start: float
    slice_end: float
    file_path: Path


@dataclass
class ChunkingResult:
    recording_id: str
    total_chunks: int
    chunks: list[ChunkPlanItem]


class AudioChunker:
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1

    @classmethod
    def compute_frame_energy(
        cls,
        audio: np.ndarray,
        sample_rate: int = 16000,
        frame_length: int = 512,
        hop_length: int = 160
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute frame-level RMS energy and dBFS curve.
        Returns (rms_values, time_seconds_per_frame).
        """
        if len(audio) == 0:
            return np.array([]), np.array([])

        pad_len = (hop_length - (len(audio) % hop_length)) % hop_length
        if pad_len > 0:
            audio_padded = np.pad(audio, (0, pad_len))
        else:
            audio_padded = audio

        rms = None
        if AudioPreprocessor._use_cuda_audio() and len(audio_padded) >= frame_length:
            try:
                import torch
                with torch.no_grad():
                    t_audio = torch.as_tensor(audio_padded, dtype=torch.float32, device="cuda")
                    t_frames = t_audio.unfold(0, frame_length, hop_length)
                    rms = torch.sqrt(torch.mean(t_frames * t_frames, dim=1) + 1e-12).cpu().numpy()
            except Exception:
                rms = None
        if rms is None:
            frames = np.lib.stride_tricks.sliding_window_view(audio_padded, frame_length)[::hop_length]
            rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        frame_times = (np.arange(len(rms)) * hop_length + frame_length // 2) / float(sample_rate)
        return rms, frame_times

    @classmethod
    def find_split_point(
        cls,
        rms: np.ndarray,
        frame_times: np.ndarray,
        search_start_sec: float,
        search_end_sec: float,
        silence_threshold_dbfs: float = SILENCE_THRESHOLD_DBFS,
        min_pause_sec: float = MIN_PAUSE_DURATION_SECONDS
    ) -> tuple[float, str, float]:
        """
        Find optimal split point in the search window [search_start_sec, search_end_sec].
        Returns (split_time_sec, strategy_used, rms_dbfs_at_split).
        """
        mask = (frame_times >= search_start_sec) & (frame_times <= search_end_sec)
        window_indices = np.where(mask)[0]

        if len(window_indices) == 0:
            # Search window fallback: midpoint
            midpoint = round((search_start_sec + search_end_sec) / 2.0, 3)
            return midpoint, "fallback_midpoint", 0.0

        window_rms = rms[window_indices]
        window_times = frame_times[window_indices]
        window_dbfs = 20.0 * np.log10(window_rms + 1e-12)

        # ---------------------------------------------------------------------
        # Priority 1: Gated Natural Silence Pause (RMS <= silence_threshold_dbfs for >= min_pause_sec)
        # ---------------------------------------------------------------------
        is_silent = (window_dbfs <= silence_threshold_dbfs)
        silent_segments: list[tuple[int, int]] = []
        in_silence = False
        start_idx = 0

        for i, silent in enumerate(is_silent):
            if silent and not in_silence:
                in_silence = True
                start_idx = i
            elif not silent and in_silence:
                in_silence = False
                silent_segments.append((start_idx, i - 1))

        if in_silence:
            silent_segments.append((start_idx, len(is_silent) - 1))

        # Filter silent segments by min_pause_sec
        valid_pauses: list[tuple[float, float, float]] = []  # (start_sec, end_sec, duration_sec)
        for s_idx, e_idx in silent_segments:
            s_time = window_times[s_idx]
            e_time = window_times[e_idx]
            dur = e_time - s_time
            if dur >= min_pause_sec:
                valid_pauses.append((s_time, e_time, dur))

        if valid_pauses:
            # Pick the longest silence pause in the window
            best_pause = max(valid_pauses, key=lambda p: p[2])
            split_sec = round((best_pause[0] + best_pause[1]) / 2.0, 3)
            # Find nearest frame dbfs
            nearest_idx = np.argmin(np.abs(window_times - split_sec))
            return split_sec, "gated_silence_pause", float(window_dbfs[nearest_idx])

        # ---------------------------------------------------------------------
        # Priority 2: RMS Valley Fallback (Lowest Energy Frame for continuous speech)
        # ---------------------------------------------------------------------
        min_idx = int(np.argmin(window_rms))
        split_sec = round(float(window_times[min_idx]), 3)
        split_dbfs = float(window_dbfs[min_idx])
        return split_sec, "rms_valley_fallback", split_dbfs

    @classmethod
    def plan_chunks(
        cls,
        audio_duration: float,
        rms: np.ndarray,
        frame_times: np.ndarray,
        target_duration: float = TARGET_CHUNK_DURATION_SECONDS,
        max_duration: float = MAX_CHUNK_DURATION_SECONDS,
        min_duration: float = MIN_CHUNK_DURATION_SECONDS,
        overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS,
        silence_threshold_dbfs: float = SILENCE_THRESHOLD_DBFS,
        min_pause_sec: float = MIN_PAUSE_DURATION_SECONDS
    ) -> list[tuple[float, float, float, float, float, float]]:
        """
        Generate continuous chunk partitioning plan.
        Returns list of (start_time, end_time, duration, overlap_start, overlap_end, slice_start, slice_end).
        """
        if audio_duration <= 0.0:
            return []

        # Short recording check: If recording is under target duration, exactly 1 chunk
        if audio_duration <= target_duration:
            return [(0.0, audio_duration, audio_duration, 0.0, 0.0, 0.0, audio_duration)]

        current_start = 0.0
        plan: list[tuple[float, float, float, float, float, float]] = []

        while current_start < audio_duration:
            remaining = audio_duration - current_start

            # If remaining duration fits in a single chunk (<= max_duration)
            if remaining <= max_duration:
                c_end = audio_duration
                c_dur = round(c_end - current_start, 3)

                # Overlaps
                ov_start = overlap_seconds if current_start > 0 else 0.0
                ov_end = 0.0
                slice_start = max(0.0, current_start - ov_start)
                slice_end = audio_duration

                plan.append((round(current_start, 3), round(c_end, 3), c_dur, ov_start, ov_end, slice_start, slice_end))
                break

            # Define search window for next split point
            # Lower bound: current_start + 20s (or target - 5s)
            # Upper bound: current_start + 28s (or target + 3s, strictly <= max_duration - 0.5s)
            search_start = current_start + max(min_duration, target_duration - 5.0)
            search_end = current_start + min(max_duration - 1.0, target_duration + 3.0)
            search_end = min(search_end, audio_duration - min_duration)

            # Find optimal split point in search window
            split_sec, strategy, _ = cls.find_split_point(
                rms=rms,
                frame_times=frame_times,
                search_start_sec=search_start,
                search_end_sec=search_end,
                silence_threshold_dbfs=silence_threshold_dbfs,
                min_pause_sec=min_pause_sec
            )

            # Ensure strict forward progress
            if split_sec <= current_start + min_duration:
                split_sec = round(current_start + target_duration, 3)

            # Trailing audio check:
            # If remaining audio after this split is < min_duration:
            tail_duration = audio_duration - split_sec
            if 0 < tail_duration < min_duration:
                # If merging into current chunk doesn't exceed max_duration, merge tail
                if (audio_duration - current_start) <= max_duration:
                    split_sec = audio_duration
                else:
                    # Otherwise adjust split earlier to leave at least min_duration for the final chunk
                    split_sec = round(audio_duration - min_duration, 3)

            c_dur = round(split_sec - current_start, 3)
            ov_start = overlap_seconds if current_start > 0 else 0.0
            ov_end = overlap_seconds if split_sec < audio_duration else 0.0
            slice_start = max(0.0, current_start - ov_start)
            slice_end = min(audio_duration, split_sec + ov_end)

            plan.append((
                round(current_start, 3),
                round(split_sec, 3),
                c_dur,
                round(current_start - slice_start, 3),
                round(slice_end - split_sec, 3),
                round(slice_start, 3),
                round(slice_end, 3)
            ))

            current_start = split_sec

        return plan

    @classmethod
    def split_recording(
        cls,
        recording_id: str,
        db: Session,
        target_duration: float = TARGET_CHUNK_DURATION_SECONDS,
        max_duration: float = MAX_CHUNK_DURATION_SECONDS,
        min_duration: float = MIN_CHUNK_DURATION_SECONDS,
        overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS,
        silence_threshold_dbfs: float = SILENCE_THRESHOLD_DBFS,
        min_pause_sec: float = MIN_PAUSE_DURATION_SECONDS,
        output_dir: Path = CHUNKS_DIR
    ) -> ChunkingResult:
        """
        Execute full intelligent chunking pipeline for a recording:
        1. Retrieve recording and ensure preprocessed audio exists.
        2. Compute frame energy and generate silence-aware chunking plan.
        3. Slice audio, save 16kHz mono 16-bit PCM WAV chunks to storage/chunks/.
        4. Persist Chunk models in database.
        5. Update Recording status to CHUNKED.
        """
        recording = db.query(Recording).filter(Recording.id == recording_id).first()
        if not recording:
            raise AudioChunkingException(f"Recording '{recording_id}' not found in database.")

        # Ensure preprocessed file exists
        if not recording.preprocessed_file_path or not Path(recording.preprocessed_file_path).exists():
            prep_res = AudioPreprocessor.process(Path(recording.file_path), recording_id)
            recording.preprocessed_file_path = str(prep_res.preprocessed_path)
            recording.sample_rate = prep_res.sample_rate
            recording.channels = prep_res.channels
            recording.duration_seconds = prep_res.total_duration
            recording.speech_duration_seconds = prep_res.speech_duration
            recording.silence_duration_seconds = prep_res.silence_duration
            recording.status = RecordingStatus.PREPROCESSED
            db.commit()
            db.refresh(recording)

        audio_path = Path(recording.preprocessed_file_path)
        audio, sr = sf.read(str(audio_path), dtype="float32")

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        total_duration = round(len(audio) / float(sr), 3)

        # 1. Compute frame energy curve
        rms, frame_times = cls.compute_frame_energy(audio, sample_rate=sr)

        # 2. Plan chunk boundaries
        raw_plan = cls.plan_chunks(
            audio_duration=total_duration,
            rms=rms,
            frame_times=frame_times,
            target_duration=target_duration,
            max_duration=max_duration,
            min_duration=min_duration,
            overlap_seconds=overlap_seconds,
            silence_threshold_dbfs=silence_threshold_dbfs,
            min_pause_sec=min_pause_sec
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        # 3. Clean up existing chunks for this recording if re-chunking
        cls.cleanup_chunks(recording_id, db)

        created_chunks: list[ChunkPlanItem] = []

        for idx, (c_start, c_end, c_dur, ov_start, ov_end, sl_start, sl_end) in enumerate(raw_plan):
            chunk_filename = f"{recording_id}_chunk_{idx:04d}.wav"
            chunk_file_path = output_dir / chunk_filename

            # Slice audio
            start_sample = int(round(sl_start * sr))
            end_sample = int(round(sl_end * sr))
            chunk_audio = audio[start_sample:end_sample]

            # Save 16kHz mono 16-bit PCM WAV
            sf.write(
                str(chunk_file_path),
                chunk_audio,
                sr,
                subtype="PCM_16"
            )

            # Create DB model
            chunk_model = Chunk(
                recording_id=recording_id,
                chunk_index=idx,
                start_time=c_start,
                end_time=c_end,
                duration=c_dur,
                file_path=str(chunk_file_path),
                overlap_start=ov_start,
                overlap_end=ov_end
            )
            db.add(chunk_model)

            created_chunks.append(ChunkPlanItem(
                chunk_index=idx,
                start_time=c_start,
                end_time=c_end,
                duration=c_dur,
                overlap_start=ov_start,
                overlap_end=ov_end,
                slice_start=sl_start,
                slice_end=sl_end,
                file_path=chunk_file_path
            ))

        recording.status = RecordingStatus.CHUNKED
        db.commit()
        db.refresh(recording)

        return ChunkingResult(
            recording_id=recording_id,
            total_chunks=len(created_chunks),
            chunks=created_chunks
        )

    @classmethod
    def cleanup_chunks(cls, recording_id: str, db: Session) -> None:
        """Remove chunk files from disk and delete database chunk rows."""
        existing_chunks = db.query(Chunk).filter(Chunk.recording_id == recording_id).all()
        for c in existing_chunks:
            fpath = Path(c.file_path)
            if fpath.exists():
                try:
                    fpath.unlink()
                except Exception:
                    pass
            db.delete(c)
        db.commit()
