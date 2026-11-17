"""
text_processing.py — REST API router for Pakistani English / Roman Urdu text processing.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Recording
from app.schemas import (
    TextProcessRequest,
    TextProcessResponse,
    TokenTagItem,
    TaggedSegmentItem,
    TaggedSegmentsResponse
)
from app.services.text_processor import TextProcessingService, TextProcessingException
from roman_urdu_nlp.detector import detect_segments, tag_tokens
from roman_urdu_nlp.normalizer import normalize

router = APIRouter(prefix="/api", tags=["Text Processing"])


@router.post(
    "/text/process",
    response_model=TextProcessResponse,
    summary="Process arbitrary Roman Urdu / Pakistani English text through full NLP pipeline"
)
def process_text_endpoint(request: TextProcessRequest):
    try:
        res = TextProcessingService.process_text(request.text)
        return TextProcessResponse(
            status="success",
            original_text=res["original_text"],
            cleaned_text=res["cleaned_text"],
            normalized_text=res["normalized_text"],
            urdu_text=res["urdu_text"],
            english_text=res["english_text"],
            token_tags=[TokenTagItem(**t) for t in res["token_tags"]],
            language_distribution=res["language_distribution"],
            stages=res["stages"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text processing failed: {str(e)}"
        )


@router.post(
    "/text/normalize",
    summary="Quickly normalize Roman Urdu spelling variants and recover ASR corruptions"
)
def normalize_text_endpoint(request: TextProcessRequest):
    try:
        normalized = normalize(request.text)
        return {
            "status": "success",
            "original_text": request.text,
            "normalized_text": normalized
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Normalization failed: {str(e)}"
        )


@router.post(
    "/text/detect",
    summary="Detect language segments and perform token-level tagging"
)
def detect_language_endpoint(request: TextProcessRequest):
    try:
        segments = detect_segments(request.text)
        token_tags = tag_tokens(request.text)

        total_valid = sum(1 for t in token_tags if t["lang"] in ("ur", "en"))
        if total_valid > 0:
            ur_count = sum(1 for t in token_tags if t["lang"] == "ur")
            en_count = sum(1 for t in token_tags if t["lang"] == "en")
            lang_dist = {
                "ur": round(ur_count / total_valid, 2),
                "en": round(en_count / total_valid, 2)
            }
        else:
            lang_dist = {"ur": 1.0}

        return {
            "status": "success",
            "text": request.text,
            "segments": [{"segment": s[0], "language": s[1]} for s in segments],
            "token_tags": [TokenTagItem(**t) for t in token_tags],
            "language_distribution": lang_dist
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Language detection failed: {str(e)}"
        )


@router.post(
    "/recordings/{recording_id}/process-text",
    summary="Batch enhance all transcript segments of a recording in the database"
)
def process_recording_text(
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
        result = TextProcessingService.process_recording_transcript(recording_id, db)
        return {
            "status": "success",
            "message": "Recording transcript enhanced successfully.",
            "data": result
        }
    except TextProcessingException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recording text processing failed: {str(e)}"
        )


@router.get(
    "/recordings/{recording_id}/segments/tagged",
    response_model=TaggedSegmentsResponse,
    summary="Get transcript segments with fine-grained token-level language metadata"
)
def get_tagged_segments_endpoint(
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
        segments = TextProcessingService.get_tagged_segments(recording_id, db)
        return TaggedSegmentsResponse(
            status="success",
            recording_id=recording_id,
            total_segments=len(segments),
            segments=[
                TaggedSegmentItem(
                    segment_id=s["segment_id"],
                    chunk_index=s["chunk_index"],
                    segment_index=s["segment_index"],
                    start_time=s["start_time"],
                    end_time=s["end_time"],
                    original_text=s["original_text"],
                    normalized_text=s["normalized_text"],
                    token_tags=[TokenTagItem(**t) for t in s["token_tags"]]
                )
                for s in segments
            ]
        )
    except TextProcessingException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch tagged segments: {str(e)}"
        )
