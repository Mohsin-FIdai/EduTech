"""
Upload router: handles file uploads, size/duration validation, and initial database registration.
"""

from fastapi import APIRouter, UploadFile, File, Form, Depends, status
from typing import Optional
from sqlalchemy.orm import Session
from pathlib import Path

from app.database import get_db
from app.models import Recording, RecordingStatus
from app.schemas import UploadSuccessResponse
from app.services.storage import StorageService
from app.services.audio_validator import AudioValidator, AudioValidationException

router = APIRouter(prefix="/api", tags=["Upload"])
storage_service = StorageService()


@router.post(
    "/upload",
    response_model=UploadSuccessResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and validate audio/video recording"
)
async def upload_recording(
    file: UploadFile = File(...),
    topic: Optional[str] = Form(None),
    recording_type: str = Form("Lecture"),
    db: Session = Depends(get_db)
):
    """
    Upload an audio or video recording.

    Accepts: .mp3, .wav, .m4a, .mp4, .webm, .mov, .ogg, .flac
    Constraints:
    - Audio <= 50 MB
    - Video <= 400 MB
    - Duration <= 30 minutes (1800s)
    """
    original_filename = file.filename or "recording.wav"

    # 1. Validate extension
    ext = AudioValidator.validate_extension(original_filename)

    # 2. Generate secure unique destination path
    recording_id, safe_filename, file_path = storage_service.generate_unique_filename(original_filename)

    try:
        # 3. Stream upload file to storage
        bytes_written = await storage_service.save_upload_file(file, file_path)

        # 4. Detect actual media type by inspecting container streams (audio-only containers classified as 'audio')
        media_type = AudioValidator.detect_media_type(file_path, ext)

        # 5. Validate file size against specific media type cap (50MB for audio, 400MB for video)
        AudioValidator.validate_size(bytes_written, media_type)

        # 6. Probe duration and file integrity
        duration_seconds = AudioValidator.probe_duration_and_integrity(file_path, media_type)

        clean_topic = topic.strip() if topic and topic.strip() else None
        clean_type = recording_type.strip() if recording_type and recording_type.strip() in ("Lecture", "Meeting") else "Lecture"

        recording = Recording(
            id=recording_id,
            filename=original_filename,
            file_path=str(file_path),
            file_format=ext,
            media_type=media_type,
            file_size_bytes=bytes_written,
            duration_seconds=duration_seconds,
            status=RecordingStatus.UPLOADED,
            topic=clean_topic,
            recording_type=clean_type
        )
        db.add(recording)
        db.commit()
        db.refresh(recording)

        return UploadSuccessResponse(
            status="success",
            message=f"Successfully uploaded and validated {media_type} recording.",
            data=recording
        )

    except AudioValidationException:
        # Clean up uploaded file if validation failed
        storage_service.delete_file(file_path)
        raise

    except Exception as e:
        # Clean up uploaded file on unexpected errors
        storage_service.delete_file(file_path)
        raise AudioValidationException(
            code="UPLOAD_FAILED",
            message=f"An unexpected error occurred during upload: {str(e)}",
            status_code=500
        )
