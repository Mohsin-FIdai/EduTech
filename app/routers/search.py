"""
app/routers/search.py — FastAPI endpoints for Vector Embeddings, Semantic Search, and Dense Q&A.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import logging

from app.database import get_db
from app.schemas import (
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchResultItem,
    SemanticQARequest,
    SemanticQAResponse,
    EmbeddingGenerationResponse,
)
from app.services.embedding_service import EmbeddingService, EmbeddingException

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/recordings",
    tags=["Semantic Search & Embeddings"]
)


from app.models import Recording, RecordingStatus

@router.post(
    "/{recording_id}/embeddings/generate",
    response_model=EmbeddingGenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate & persist dense vector embeddings for recording transcript segments"
)
def generate_embeddings(recording_id: str, db: Session = Depends(get_db)):
    """Compute and persist 384-dimensional vector embeddings for all transcript segments."""
    try:
        count = EmbeddingService.generate_and_store_embeddings(recording_id, db)
        rec = db.query(Recording).filter(Recording.id == recording_id).first()
        if rec:
            rec.status = RecordingStatus.COMPLETED
            rec.error_message = None
            db.commit()
        return EmbeddingGenerationResponse(
            status="success",
            recording_id=recording_id,
            total_segments_embedded=count,
            dimensions=384
        )
    except EmbeddingException as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in generate_embeddings: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    "/{recording_id}/search/semantic",
    response_model=SemanticSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Perform dense semantic search across transcript segments"
)
def semantic_search(recording_id: str, req: SemanticSearchRequest, db: Session = Depends(get_db)):
    """Search transcript segments using dense semantic similarity with top-k ranking and threshold filtering."""
    try:
        results = EmbeddingService.search_semantic(
            recording_id=recording_id,
            query=req.query,
            top_k=req.top_k,
            threshold=req.threshold,
            db=db
        )
        items = [SemanticSearchResultItem(**r) for r in results]
        return SemanticSearchResponse(
            status="success",
            recording_id=recording_id,
            query=req.query,
            total_results=len(items),
            results=items
        )
    except EmbeddingException as e:
        logger.error(f"Semantic search error: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in semantic_search: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    "/{recording_id}/qa-semantic",
    response_model=SemanticQAResponse,
    status_code=status.HTTP_200_OK,
    summary="Answer questions grounded in transcript segments using dense vector similarity"
)
def semantic_qa(recording_id: str, req: SemanticQARequest, db: Session = Depends(get_db)):
    """Grounded Q&A with paraphrase/synonym resolution and source timestamp citations."""
    try:
        ans = EmbeddingService.answer_semantic_qa(
            recording_id=recording_id,
            question=req.question,
            threshold=req.threshold,
            db=db
        )
        return SemanticQAResponse(**ans)
    except EmbeddingException as e:
        logger.error(f"Semantic QA error: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in semantic_qa: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
