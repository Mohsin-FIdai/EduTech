"""
Recordings router: list, retrieve, and delete recording metadata and media.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session
from pathlib import Path
import re

from app.database import get_db
from fastapi.responses import FileResponse, StreamingResponse
from app.models import Recording, RecordingStatus, Chunk, Transcript, TranscriptSegment
from app.schemas import (
    RecordingResponse,
    RecordingListResponse,
    PreprocessingResponse,
    ChunkItemResponse,
    ChunkListResponse,
    ChunkingResponse,
    ChunkingRequest,
    TranscriptSegmentResponse,
    TranscriptResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    SegmentCorrectionRequest,
    SegmentCorrectionResponse
)
from app.services.storage import StorageService
from app.services.audio_chunker import AudioChunker, AudioChunkingException
from app.services.transcription_service import TranscriptionService, TranscriptionException

router = APIRouter(prefix="/api/recordings", tags=["Recordings"])
storage_service = StorageService()


@router.get(
    "",
    response_model=RecordingListResponse,
    summary="List all uploaded recordings"
)
def list_recordings(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    recordings = db.query(Recording).order_by(Recording.created_at.desc()).offset(skip).limit(limit).all()
    total = db.query(Recording).count()
    return RecordingListResponse(
        status="success",
        total=total,
        data=recordings
    )


@router.get(
    "/{recording_id}",
    response_model=RecordingResponse,
    summary="Get single recording metadata"
)
def get_recording(
    recording_id: str,
    db: Session = Depends(get_db)
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{recording_id}' not found."
        )
    return recording


@router.post(
    "/{recording_id}/preprocess",
    response_model=PreprocessingResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract audio, resample to 16kHz mono, normalize volume, and detect silence intervals"
)
def preprocess_recording(
    recording_id: str,
    db: Session = Depends(get_db)
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{recording_id}' not found."
        )

    from app.services.audio_preprocessor import AudioPreprocessor, AudioPreprocessingException

    input_path = Path(recording.file_path)
    if not input_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Uploaded source file '{recording.file_path}' not found on storage disk."
        )

    try:
        result = AudioPreprocessor.process(input_path, recording_id)

        # Update recording metadata in database
        recording.preprocessed_file_path = str(result.preprocessed_path)
        recording.sample_rate = result.sample_rate
        recording.channels = result.channels
        recording.duration_seconds = result.total_duration
        recording.speech_duration_seconds = result.speech_duration
        recording.silence_duration_seconds = result.silence_duration
        recording.status = RecordingStatus.PREPROCESSED
        recording.error_message = None

        db.commit()
        db.refresh(recording)

        return PreprocessingResponse(
            status="success",
            message="Audio extracted, resampled to 16kHz mono, volume-normalized, and silence-detected successfully.",
            recording_id=recording.id,
            preprocessed_file_path=recording.preprocessed_file_path,
            sample_rate=result.sample_rate,
            channels=result.channels,
            total_duration_seconds=result.total_duration,
            speech_duration_seconds=result.speech_duration,
            silence_duration_seconds=result.silence_duration,
            speech_intervals=result.speech_intervals
        )
    except AudioPreprocessingException as e:
        recording.status = RecordingStatus.FAILED
        recording.error_message = str(e.message)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing failed: {e.message}"
        )


@router.post(
    "/{recording_id}/chunk",
    response_model=ChunkingResponse,
    status_code=status.HTTP_200_OK,
    summary="Partition preprocessed audio into intelligent, silence-aware chunks"
)
def chunk_recording(
    recording_id: str,
    request: ChunkingRequest = ChunkingRequest(),
    db: Session = Depends(get_db)
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{recording_id}' not found."
        )

    try:
        result = AudioChunker.split_recording(
            recording_id=recording_id,
            db=db,
            target_duration=request.target_duration or 25.0,
            max_duration=request.max_duration or 30.0,
            min_duration=request.min_duration or 2.0,
            overlap_seconds=request.overlap_seconds or 0.5
        )

        db_chunks = db.query(Chunk).filter(Chunk.recording_id == recording_id).order_by(Chunk.chunk_index.asc()).all()

        return ChunkingResponse(
            status="success",
            message="Audio partitioned into intelligent chunks successfully.",
            recording_id=recording_id,
            total_chunks=len(db_chunks),
            chunks=[ChunkItemResponse.model_validate(c) for c in db_chunks]
        )
    except AudioChunkingException as e:
        recording.status = RecordingStatus.FAILED
        recording.error_message = str(e.message)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunking failed: {e.message}"
        )


@router.get(
    "/{recording_id}/chunks",
    response_model=ChunkListResponse,
    summary="Get all chunks for a recording in chronological order"
)
def get_recording_chunks(
    recording_id: str,
    db: Session = Depends(get_db)
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{recording_id}' not found."
        )

    chunks = db.query(Chunk).filter(Chunk.recording_id == recording_id).order_by(Chunk.chunk_index.asc()).all()
    return ChunkListResponse(
        status="success",
        recording_id=recording_id,
        total_chunks=len(chunks),
        data=[ChunkItemResponse.model_validate(c) for c in chunks]
    )


def _range_audio_response(target_path: Path, media_type: str, request: Request) -> Response:
    """Serve an audio file with HTTP Range (206 Partial Content) support for browser seeking."""
    file_size = target_path.stat().st_size
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(
            path=str(target_path),
            media_type=media_type,
            filename=target_path.name,
            headers={"Accept-Ranges": "bytes"}
        )

    match = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match:
        return Response(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{file_size}"}
        )

    start_str, end_str = match.groups()
    if start_str == "" and end_str == "":
        return Response(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{file_size}"}
        )

    if start_str == "":
        suffix_len = int(end_str)
        start = max(0, file_size - suffix_len)
        end = file_size - 1
    else:
        start = int(start_str)
        end = int(end_str) if end_str != "" else file_size - 1

    if file_size == 0 or start >= file_size or start > end:
        return Response(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{file_size}"}
        )

    end = min(end, file_size - 1)
    chunk_len = end - start + 1

    def iter_range(path: Path, offset: int, length: int, chunk_size: int = 65536):
        with open(path, "rb") as f:
            f.seek(offset)
            remaining = length
            while remaining > 0:
                data = f.read(min(chunk_size, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_len),
        "Content-Disposition": f'inline; filename="{target_path.name}"',
    }
    return StreamingResponse(
        iter_range(target_path, start, chunk_len),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        media_type=media_type,
        headers=headers,
    )


@router.get(
    "/{recording_id}/audio",
    summary="Stream master audio or preprocessed WAV for a recording"
)
def stream_recording_audio(
    recording_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{recording_id}' not found."
        )

    # Prefer preprocessed 16kHz WAV if available, otherwise original uploaded file
    target_path = None
    if recording.preprocessed_file_path and Path(recording.preprocessed_file_path).exists():
        target_path = Path(recording.preprocessed_file_path)
    elif recording.file_path and Path(recording.file_path).exists():
        target_path = Path(recording.file_path)

    if not target_path or not target_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file for recording '{recording_id}' not found on disk."
        )

    ext = target_path.suffix.lower()
    media_type = "audio/wav"
    if ext == ".mp3":
        media_type = "audio/mpeg"
    elif ext == ".m4a":
        media_type = "audio/mp4"
    elif ext == ".ogg":
        media_type = "audio/ogg"
    elif ext == ".flac":
        media_type = "audio/flac"
    elif ext in (".mp4", ".webm", ".mov"):
        media_type = f"video/{ext.replace('.', '')}"

    return _range_audio_response(target_path, media_type, request)


@router.get(
    "/{recording_id}/chunks/{chunk_index}/audio",
    summary="Stream or download a specific chunk WAV audio file"
)
def stream_chunk_audio(
    recording_id: str,
    chunk_index: int,
    request: Request,
    db: Session = Depends(get_db)
):
    chunk = db.query(Chunk).filter(
        Chunk.recording_id == recording_id,
        Chunk.chunk_index == chunk_index
    ).first()

    if not chunk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk index {chunk_index} for recording '{recording_id}' not found."
        )

    fpath = Path(chunk.file_path)
    if not fpath.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk file at '{chunk.file_path}' does not exist on disk."
        )

    return _range_audio_response(fpath, "audio/wav", request)


@router.post(
    "/{recording_id}/transcribe",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_200_OK,
    summary="Transcribe recording audio chunks with Whisper into globally timestamped transcript"
)
def transcribe_recording(
    recording_id: str,
    request: TranscriptionRequest = TranscriptionRequest(),
    db: Session = Depends(get_db)
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{recording_id}' not found."
        )

    try:
        result = TranscriptionService.transcribe_recording(
            recording_id=recording_id,
            db=db,
            model_size=request.model_size or "base",
            language=request.language,
            temperature=request.temperature if request.temperature is not None else 0.0,
            initial_prompt=request.topic
        )

        db_transcript = db.query(Transcript).filter(Transcript.id == result.transcript_id).first()
        db_segments = db.query(TranscriptSegment).filter(
            TranscriptSegment.transcript_id == result.transcript_id
        ).order_by(TranscriptSegment.segment_index.asc()).all()

        return TranscriptionResponse(
            status="success",
            message="Recording transcribed successfully.",
            recording_id=recording_id,
            transcript_id=result.transcript_id,
            total_segments=len(db_segments),
            total_words=result.total_words,
            language_summary=result.language_summary,
            original_transcript=result.original_transcript,
            normalized_transcript=result.normalized_transcript,
            urdu_transcript=getattr(db_transcript, "urdu_transcript", None),
            english_transcript=getattr(db_transcript, "english_transcript", None),
            segments=[TranscriptSegmentResponse.model_validate(s) for s in db_segments]
        )
    except TranscriptionException as e:
        recording.status = RecordingStatus.FAILED
        recording.error_message = str(e.message)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {e.message}"
        )
    except Exception as e:
        try:
            db.rollback()
            recording = db.query(Recording).filter(Recording.id == recording_id).first()
            if recording:
                recording.status = RecordingStatus.FAILED
                recording.error_message = str(e)
                db.commit()
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )


@router.get(
    "/{recording_id}/transcript",
    response_model=TranscriptResponse,
    summary="Get full structured transcript with global segments for a recording"
)
def get_recording_transcript(
    recording_id: str,
    db: Session = Depends(get_db)
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{recording_id}' not found."
        )

    transcript = db.query(Transcript).filter(Transcript.recording_id == recording_id).first()
    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcript for recording '{recording_id}' not found. Please transcribe the recording first."
        )

    segments = db.query(TranscriptSegment).filter(
        TranscriptSegment.transcript_id == transcript.id
    ).order_by(TranscriptSegment.segment_index.asc()).all()

    total_words = len(transcript.original_transcript.split()) if transcript.original_transcript else 0

    return TranscriptResponse(
        id=transcript.id,
        recording_id=recording_id,
        original_transcript=transcript.original_transcript,
        normalized_transcript=transcript.normalized_transcript,
        urdu_transcript=getattr(transcript, "urdu_transcript", None),
        english_transcript=getattr(transcript, "english_transcript", None),
        language_summary=transcript.language_summary,
        total_words=total_words,
        total_segments=len(segments),
        segments=[TranscriptSegmentResponse.model_validate(s) for s in segments],
        created_at=transcript.created_at
    )


@router.post(
    "/{recording_id}/contextual-refine",
    summary="Apply high-precision context-aware semantic refinement to existing recording transcript"
)
def contextual_refine_transcript(
    recording_id: str,
    db: Session = Depends(get_db)
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{recording_id}' not found."
        )
    try:
        summary = TranscriptionService.refine_existing_transcription(recording_id, db)
        return {
            "status": "success",
            "message": "Contextual refinement completed successfully.",
            **summary
        }
    except TranscriptionException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Contextual refinement failed: {str(e)}"
        )


@router.post(
    "/{recording_id}/segments/{segment_id}/correct",
    response_model=SegmentCorrectionResponse,
    summary="Manually correct a transcript segment and propagate recomputations downstream"
)
def correct_segment_endpoint(
    recording_id: str,
    segment_id: str,
    request: SegmentCorrectionRequest,
    db: Session = Depends(get_db)
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording '{recording_id}' not found."
        )

    try:
        result = TranscriptionService.update_segment_correction(
            recording_id=recording_id,
            segment_id=segment_id,
            db=db,
            normalized_text=request.normalized_text,
            text=request.text,
            scope=request.scope or "full"
        )
        return SegmentCorrectionResponse(**result)
    except TranscriptionException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segment correction failed: {str(e)}"
        )


@router.get(
    "/{recording_id}/chunks/{chunk_index}/transcript",
    summary="Get transcript segments for a specific chunk index"
)
def get_chunk_transcript(
    recording_id: str,
    chunk_index: int,
    db: Session = Depends(get_db)
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{recording_id}' not found."
        )

    transcript = db.query(Transcript).filter(Transcript.recording_id == recording_id).first()
    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcript for recording '{recording_id}' not found."
        )

    segments = db.query(TranscriptSegment).filter(
        TranscriptSegment.transcript_id == transcript.id,
        TranscriptSegment.chunk_index == chunk_index
    ).order_by(TranscriptSegment.segment_index.asc()).all()

    chunk_text = " ".join([s.text for s in segments]).strip()

    return {
        "status": "success",
        "recording_id": recording_id,
        "chunk_index": chunk_index,
        "text": chunk_text,
        "total_segments": len(segments),
        "segments": [TranscriptSegmentResponse.model_validate(s) for s in segments]
    }


@router.post(
    "/{recording_id}/process",
    summary="Execute full end-to-end intelligence pipeline on a recording"
)
def process_full_recording(
    recording_id: str,
    db: Session = Depends(get_db)
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{recording_id}' not found."
        )

    from app.services.audio_preprocessor import AudioPreprocessor
    from app.services.audio_chunker import AudioChunker
    from app.services.transcription_service import TranscriptionService
    from app.services.intelligence_service import LectureIntelligenceService
    from app.services.embedding_service import EmbeddingService

    try:
        # Step 1: Preprocess
        input_path = Path(recording.file_path)
        if not input_path.exists():
            raise HTTPException(status_code=404, detail="Source file not found on disk.")
        prep_result = AudioPreprocessor.process(input_path, recording_id)
        recording.preprocessed_file_path = str(prep_result.preprocessed_path)
        recording.sample_rate = prep_result.sample_rate
        recording.channels = prep_result.channels
        recording.duration_seconds = prep_result.total_duration
        recording.speech_duration_seconds = prep_result.speech_duration
        recording.silence_duration_seconds = prep_result.silence_duration
        recording.status = RecordingStatus.PREPROCESSED
        db.commit()

        # Step 2: Intelligent Chunking
        AudioChunker.split_recording(recording_id=recording_id, db=db)
        recording.status = RecordingStatus.CHUNKED
        db.commit()

        # Step 3: Whisper Transcription & Normalization
        TranscriptionService.transcribe_recording(recording_id=recording_id, db=db)
        recording.status = RecordingStatus.TRANSCRIBED
        db.commit()

        # Step 4: Extract Structured Intelligence (Notes, Tasks, Quizzes)
        LectureIntelligenceService.generate_intelligence_report(recording_id=recording_id, db=db)

        # Step 5: Generate Dense Vector Embeddings for Q&A
        try:
            EmbeddingService.generate_and_store_embeddings(recording_id=recording_id, db=db)
        except Exception:
            pass

        recording.status = RecordingStatus.COMPLETED
        recording.error_message = None
        db.commit()
        db.refresh(recording)

        return {
            "status": "success",
            "message": "Recording processed through full intelligence pipeline successfully.",
            "recording_id": recording_id,
            "data": RecordingResponse.model_validate(recording)
        }
    except Exception as e:
        recording.status = RecordingStatus.FAILED
        recording.error_message = str(e)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing failed: {str(e)}"
        )


@router.delete(
    "/{recording_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a recording, its chunks, transcripts, and associated storage files"
)
def delete_recording(
    recording_id: str,
    db: Session = Depends(get_db)
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{recording_id}' not found."
        )

    # Clean up physical chunk files
    AudioChunker.cleanup_chunks(recording_id, db)

    # Clean up DB transcripts
    TranscriptionService.cleanup_transcripts(recording_id, db)

    # Delete physical source and preprocessed files
    if recording.file_path:
        storage_service.delete_file(recording.file_path)
    if recording.preprocessed_file_path:
        storage_service.delete_file(recording.preprocessed_file_path)

    # Delete DB record (cascades to chunks/transcripts)
    db.delete(recording)
    db.commit()

    return {"status": "success", "message": f"Recording '{recording_id}' deleted successfully."}
