"""
Audio Extraction and Preprocessing service:
- Video-to-audio extraction
- Resampling to 16,000 Hz (16 kHz)
- Downmixing to mono (1 channel)
- Loudness and peak volume normalization
- Silence and speech interval detection
"""

from pathlib import Path
import subprocess
from dataclasses import dataclass
import numpy as np
import soundfile as sf
import imageio_ffmpeg

from app.config import PROCESSED_DIR
from app.config import AUDIO_TARGET_PEAK_NORM, AUDIO_DENOISE_PROP_DECREASE, VAD_TOP_DB, VAD_MIN_SPEECH_DURATION_SEC


@dataclass
class PreprocessingResult:
    preprocessed_path: Path
    sample_rate: int
    channels: int
    total_duration: float
    speech_duration: float
    silence_duration: float
    speech_intervals: list[tuple[float, float]]
    rms_energy: float
    max_peak: float


class AudioPreprocessingException(Exception):
    """Exception raised during audio extraction and preprocessing."""
    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class AudioPreprocessor:
    TARGET_SAMPLE_RATE: int = 16000
    TARGET_CHANNELS: int = 1
    TARGET_PEAK_NORM: float = AUDIO_TARGET_PEAK_NORM  # -0.5 dBFS

    @staticmethod
    def _use_cuda_audio() -> bool:
        try:
            from app.config import AUDIO_DEVICE
            import torch
            return AUDIO_DEVICE != "cpu" and torch.cuda.is_available()
        except Exception:
            return False

    @classmethod
    def extract_to_raw_pcm_wav(cls, input_path: Path, temp_wav_path: Path) -> None:
        """
        Extract audio stream, convert to 16kHz mono 16-bit PCM WAV using FFmpeg.
        Uses hardware acceleration (-hwaccel auto) when CUDA is available, with automatic CPU fallback.
        """
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        base_args = [
            "-i", str(input_path),
            "-vn",                       # Strip video
            "-acodec", "pcm_s16le",      # 16-bit PCM
            "-ar", str(cls.TARGET_SAMPLE_RATE),  # 16 kHz
            "-ac", str(cls.TARGET_CHANNELS),     # Mono
            str(temp_wav_path)
        ]

        if cls._use_cuda_audio():
            hw_cmd = [ffmpeg_exe, "-y", "-hwaccel", "auto"] + base_args
            proc = subprocess.run(hw_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode == 0 and temp_wav_path.exists():
                return

        cmd = [ffmpeg_exe, "-y"] + base_args
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if proc.returncode != 0:
            raise AudioPreprocessingException(
                message=f"FFmpeg audio extraction failed for '{input_path.name}'.",
                details={"stderr": proc.stderr}
            )

    @classmethod
    def denoise_audio(
        cls,
        audio: np.ndarray,
        sample_rate: int = 16000,
        prop_decrease: float = AUDIO_DENOISE_PROP_DECREASE
    ) -> np.ndarray:
        """
        Spectral gating noise reduction with CUDA GPU STFT/iSTFT acceleration (and SciPy CPU fallback).
        Suppresses stationary background noise (fan hum, HVAC, ambient room hiss)
        without degrading speech intelligibility or introducing musical noise.
        """
        if len(audio) == 0:
            return audio
        if cls._use_cuda_audio():
            try:
                import torch
                with torch.no_grad():
                    x = torch.as_tensor(audio, dtype=torch.float32, device="cuda")
                    n_fft = 1024
                    hop_length = 256
                    win = torch.hann_window(n_fft, device="cuda")
                    Zxx = torch.stft(
                        x,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=n_fft,
                        window=win,
                        center=True,
                        return_complex=True,
                    )
                    magnitude = torch.abs(Zxx)
                    # Estimate stationary noise floor from lowest 10% energy frames on GPU
                    noise_floor = torch.quantile(magnitude, 0.10, dim=-1, keepdim=True)
                    gain = torch.clamp(1.0 - (prop_decrease * noise_floor / (magnitude + 1e-8)), min=0.0)
                    Zxx_denoised = Zxx * gain
                    denoised_t = torch.istft(
                        Zxx_denoised,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=n_fft,
                        window=win,
                        center=True,
                        length=len(audio),
                    )
                    return denoised_t.cpu().numpy().astype(np.float32)
            except Exception:
                pass
        try:
            from scipy import signal
            f, t, Zxx = signal.stft(audio, fs=sample_rate, nperseg=1024, noverlap=768)
            magnitude = np.abs(Zxx)
            phase = np.angle(Zxx)
            # Estimate stationary noise floor from the lowest 10% energy frames
            noise_floor = np.percentile(magnitude, 10, axis=-1, keepdims=True)
            gain = np.maximum(0.0, 1.0 - (prop_decrease * noise_floor / (magnitude + 1e-8)))
            Zxx_denoised = gain * magnitude * np.exp(1j * phase)
            _, denoised = signal.istft(Zxx_denoised, fs=sample_rate, nperseg=1024, noverlap=768)
            if len(denoised) > len(audio):
                denoised = denoised[:len(audio)]
            elif len(denoised) < len(audio):
                denoised = np.pad(denoised, (0, len(audio) - len(denoised)))
            return denoised.astype(np.float32)
        except Exception:
            return audio

    @classmethod
    def normalize_audio(cls, audio: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Normalize audio signal volume to prevent clipping and balance levels.
        Uses CUDA GPU tensor reduction when available, with NumPy fallback.
        Returns (normalized_audio, rms_energy, max_peak).
        """
        if len(audio) == 0:
            return audio, 0.0, 0.0

        if cls._use_cuda_audio() and len(audio) >= 16000:
            try:
                import torch
                with torch.no_grad():
                    x = torch.as_tensor(audio, dtype=torch.float32, device="cuda")
                    max_abs = float(torch.max(torch.abs(x)).item())
                    if max_abs > 0:
                        x = x * (cls.TARGET_PEAK_NORM / max_abs)
                    rms = float(torch.sqrt(torch.mean(x * x)).item())
                    peak = float(torch.max(torch.abs(x)).item())
                    return x.cpu().numpy().astype(np.float32), rms, peak
            except Exception:
                pass

        max_abs = np.max(np.abs(audio)) if len(audio) > 0 else 0.0

        if max_abs > 0:
            # Scale peak to TARGET_PEAK_NORM (0.95)
            normalized = audio * (cls.TARGET_PEAK_NORM / max_abs)
        else:
            normalized = audio

        rms = float(np.sqrt(np.mean(normalized ** 2))) if len(normalized) > 0 else 0.0
        peak = float(np.max(np.abs(normalized))) if len(normalized) > 0 else 0.0
        return normalized, rms, peak

    @classmethod
    def detect_speech_intervals(
        cls,
        audio: np.ndarray,
        sample_rate: int = 16000,
        top_db: float = VAD_TOP_DB,
        min_speech_duration_sec: float = VAD_MIN_SPEECH_DURATION_SEC
    ) -> tuple[list[tuple[float, float]], float, float]:
        """
        Detect non-silent speech segments and compute speech vs silence durations
        using sliding window frame-energy thresholding with minimum segment filtering.
        Returns (speech_intervals_in_seconds, total_speech_seconds, total_silence_seconds).
        """
        if len(audio) == 0:
            return [], 0.0, 0.0

        total_duration = round(len(audio) / float(sample_rate), 3)
        frame_length = 512
        hop_length = 160  # 10ms hop

        # Pad audio to integer multiple of hop_length
        pad_len = (hop_length - (len(audio) % hop_length)) % hop_length
        if pad_len > 0:
            audio_padded = np.pad(audio, (0, pad_len))
        else:
            audio_padded = audio

        # Sliding window frame extraction (CUDA GPU accelerated when available)
        rms = None
        if cls._use_cuda_audio() and len(audio_padded) >= frame_length:
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

        max_rms = np.max(rms) if len(rms) > 0 else 0.0
        if max_rms <= 1e-6:
            return [], 0.0, total_duration

        threshold = max_rms * (10.0 ** (-top_db / 20.0))
        is_speech = (rms >= threshold)

        raw_intervals: list[tuple[float, float]] = []
        in_speech = False
        start_frame = 0

        for i, active in enumerate(is_speech):
            if active and not in_speech:
                in_speech = True
                start_frame = i
            elif not active and in_speech:
                in_speech = False
                start_sec = round((start_frame * hop_length) / sample_rate, 3)
                end_sec = round((i * hop_length + frame_length) / sample_rate, 3)
                end_sec = min(end_sec, total_duration)
                if end_sec > start_sec:
                    raw_intervals.append((start_sec, end_sec))

        if in_speech:
            start_sec = round((start_frame * hop_length) / sample_rate, 3)
            end_sec = total_duration
            if end_sec > start_sec:
                raw_intervals.append((start_sec, end_sec))

        # Filter out micro-clicks / transients shorter than min_speech_duration_sec
        intervals = [
            iv for iv in raw_intervals
            if (iv[1] - iv[0]) >= min_speech_duration_sec
        ]

        total_speech = round(sum(end - start for start, end in intervals), 3)
        total_speech = min(total_speech, total_duration)
        total_silence = round(max(0.0, total_duration - total_speech), 3)

        return intervals, total_speech, total_silence

    @classmethod
    def process(
        cls,
        input_path: Path,
        recording_id: str,
        output_dir: Path = PROCESSED_DIR,
        apply_denoise: bool = True
    ) -> PreprocessingResult:
        """
        Full extraction and preprocessing pipeline:
        1. Extract and resample to 16kHz mono WAV via FFmpeg.
        2. Apply spectral gating noise reduction.
        3. Apply volume and peak loudness normalization.
        4. Detect silence and speech intervals.
        5. Save master 16-bit PCM WAV to storage/processed/.
        """
        if not input_path.exists():
            raise AudioPreprocessingException(f"Input file not found at '{input_path}'.")

        output_dir.mkdir(parents=True, exist_ok=True)
        final_processed_path = output_dir / f"{recording_id}_preprocessed.wav"
        temp_pcm_path = output_dir / f"{recording_id}_temp_raw.wav"

        try:
            # 1. Extract audio via FFmpeg
            cls.extract_to_raw_pcm_wav(input_path, temp_pcm_path)

            # 2. Read PCM data into numpy array
            audio, sr = sf.read(str(temp_pcm_path), dtype="float32")

            # Ensure 1D array (mono)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            if sr != cls.TARGET_SAMPLE_RATE:
                from scipy import signal
                num_samples = int(len(audio) * float(cls.TARGET_SAMPLE_RATE) / sr)
                audio = signal.resample(audio, num_samples).astype(np.float32)
                sr = cls.TARGET_SAMPLE_RATE

            # 3. Apply lightweight spectral noise reduction
            if apply_denoise:
                audio = cls.denoise_audio(audio, sample_rate=cls.TARGET_SAMPLE_RATE, prop_decrease=0.85)

            # 4. Normalize volume
            norm_audio, rms_energy, max_peak = cls.normalize_audio(audio)

            # 5. Detect silence & speech intervals
            speech_intervals, speech_dur, silence_dur = cls.detect_speech_intervals(
                norm_audio,
                sample_rate=cls.TARGET_SAMPLE_RATE,
                top_db=20.0
            )

            # 6. Write preprocessed master WAV (16-bit PCM, 16kHz, mono)
            sf.write(
                str(final_processed_path),
                norm_audio,
                cls.TARGET_SAMPLE_RATE,
                subtype="PCM_16"
            )

            total_duration = round(len(norm_audio) / float(cls.TARGET_SAMPLE_RATE), 3)

            return PreprocessingResult(
                preprocessed_path=final_processed_path,
                sample_rate=cls.TARGET_SAMPLE_RATE,
                channels=cls.TARGET_CHANNELS,
                total_duration=total_duration,
                speech_duration=speech_dur,
                silence_duration=silence_dur,
                speech_intervals=speech_intervals,
                rms_energy=rms_energy,
                max_peak=max_peak
            )

        finally:
            # Clean up temporary raw WAV if created
            if temp_pcm_path.exists():
                temp_pcm_path.unlink()
