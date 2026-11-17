"""
Audio and video validation service: format, header, size, duration, and corruption detection.
"""

from pathlib import Path
import subprocess
import json
import wave
from tinytag import TinyTag
import mutagen
import imageio_ffmpeg

from app.config import (
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    ALL_SUPPORTED_EXTENSIONS,
    MAX_AUDIO_SIZE_BYTES,
    MAX_VIDEO_SIZE_BYTES,
    MAX_DURATION_SECONDS,
)


class AudioValidationException(Exception):
    """Base validation error."""
    def __init__(self, code: str, message: str, status_code: int = 400, details: dict | None = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class UnsupportedFormatException(AudioValidationException):
    def __init__(self, extension: str, allowed: set[str]):
        super().__init__(
            code="UNSUPPORTED_FORMAT",
            message=f"Unsupported file format '{extension}'. Supported formats: {', '.join(sorted(allowed))}",
            status_code=415,
            details={"received_extension": extension, "allowed_extensions": sorted(list(allowed))}
        )


class OversizedFileException(AudioValidationException):
    def __init__(self, size_bytes: int, max_bytes: int, media_type: str):
        size_mb = size_bytes / (1024 * 1024)
        max_mb = max_bytes / (1024 * 1024)
        super().__init__(
            code="FILE_TOO_LARGE",
            message=f"{media_type.capitalize()} file size ({size_mb:.2f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB.",
            status_code=413,
            details={"size_bytes": size_bytes, "max_bytes": max_bytes, "media_type": media_type}
        )


class CorruptFileException(AudioValidationException):
    def __init__(self, reason: str = "Invalid or unreadable audio/video stream header"):
        super().__init__(
            code="CORRUPT_FILE",
            message=f"File validation failed: {reason}",
            status_code=422,
            details={"reason": reason}
        )


class DurationExceededException(AudioValidationException):
    def __init__(self, duration_seconds: float, max_seconds: float):
        duration_min = duration_seconds / 60
        max_min = max_seconds / 60
        super().__init__(
            code="DURATION_EXCEEDED",
            message=f"Recording duration ({duration_min:.2f} min / {duration_seconds:.1f}s) exceeds maximum allowed limit of {max_min:.0f} min ({max_seconds:.0f}s).",
            status_code=400,
            details={"duration_seconds": duration_seconds, "max_seconds": max_seconds}
        )


class AudioValidator:
    @staticmethod
    def validate_extension(filename: str) -> str:
        """Validate extension against all supported audio/video formats."""
        if not filename:
            raise UnsupportedFormatException(extension="empty", allowed=ALL_SUPPORTED_EXTENSIONS)

        ext = Path(filename).suffix.lower()
        if ext not in ALL_SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatException(extension=ext or "none", allowed=ALL_SUPPORTED_EXTENSIONS)
        return ext

    @classmethod
    def detect_media_type(cls, file_path: Path, ext: str) -> str:
        """
        Determine actual media_type ('audio' vs 'video').
        For video extensions (.mp4, .webm, .mov), inspect whether video streams exist.
        Audio-only containers (e.g. voice memo webm/mp4) are classified as 'audio'
        and subject to the 50 MB audio cap.
        """
        if ext in SUPPORTED_AUDIO_EXTENSIONS:
            return "audio"

        # For video container formats, check if video stream is present
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [ffmpeg_exe, "-i", str(file_path)]
            proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, timeout=5)
            output = proc.stderr

            has_video = any("Stream #" in line and "Video:" in line for line in output.splitlines())
            return "video" if has_video else "audio"
        except Exception:
            # Default to video if probe fails for video extensions
            return "video"

    @staticmethod
    def validate_size(size_bytes: int, media_type: str) -> None:
        """
        Validate file size against audio (50MB) and video (400MB) limits.
        """
        if size_bytes <= 0:
            raise CorruptFileException(reason="File is empty (0 bytes)")

        max_allowed = MAX_AUDIO_SIZE_BYTES if media_type == "audio" else MAX_VIDEO_SIZE_BYTES
        if size_bytes > max_allowed:
            raise OversizedFileException(
                size_bytes=size_bytes,
                max_bytes=max_allowed,
                media_type=media_type
            )

    @staticmethod
    def _probe_with_ffmpeg(file_path: Path) -> float | None:
        """Fallback to bundled ffmpeg / ffprobe for any container."""
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [ffmpeg_exe, "-i", str(file_path)]
            proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, timeout=5)
            output = proc.stderr

            # Look for "Duration: 00:01:23.45"
            for line in output.splitlines():
                if "Duration:" in line:
                    parts = line.split("Duration:")[1].split(",")[0].strip()
                    h, m, s = parts.split(":")
                    duration = float(h) * 3600 + float(m) * 60 + float(s)
                    if duration > 0:
                        return duration
        except Exception:
            pass
        return None

    @classmethod
    def probe_duration_and_integrity(cls, file_path: Path, media_type: str) -> float:
        """
        Inspect file headers, probe duration, and detect corruption.
        Returns duration in seconds.
        """
        if not file_path.exists() or file_path.stat().st_size == 0:
            raise CorruptFileException(reason="File does not exist or is empty")

        duration: float | None = None
        ext = file_path.suffix.lower()

        # 1. Try standard wave module for .wav
        if ext == ".wav":
            try:
                with wave.open(str(file_path), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    if rate > 0:
                        duration = frames / float(rate)
            except Exception:
                pass

        # 2. Try TinyTag
        if duration is None:
            try:
                tag = TinyTag.get(str(file_path))
                if tag.duration is not None and tag.duration > 0:
                    duration = float(tag.duration)
            except Exception:
                pass

        # 3. Try Mutagen
        if duration is None:
            try:
                mf = mutagen.File(str(file_path))
                if mf is not None and mf.info and getattr(mf.info, "length", None):
                    duration = float(mf.info.length)
            except Exception:
                pass

        # 4. Fallback to bundled ffmpeg probe
        if duration is None:
            duration = cls._probe_with_ffmpeg(file_path)

        if duration is None or duration <= 0:
            raise CorruptFileException(
                reason=f"Unable to parse valid {media_type} streams from the uploaded file (corrupt or incomplete headers)"
            )

        # Validate max duration (30 mins / 1800s)
        if duration > MAX_DURATION_SECONDS:
            raise DurationExceededException(
                duration_seconds=duration,
                max_seconds=MAX_DURATION_SECONDS
            )

        return duration
