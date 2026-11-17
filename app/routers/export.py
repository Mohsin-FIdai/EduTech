"""
export.py — REST API router for exporting Lecture & Meeting Intelligence reports and transcripts.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.export_service import ExportService, ExportException

router = APIRouter(prefix="/api/recordings", tags=["Export"])


@router.get(
    "/{recording_id}/export",
    summary="Export generated lecture & meeting intelligence report or transcript in PDF, DOCX, TXT, or Markdown format"
)
def export_recording_endpoint(
    recording_id: str,
    content_type: str = Query(
        "full",
        description="Content selection: 'full', 'transcript', 'summary', 'notes', 'assignments', 'action_items', 'quizzes'"
    ),
    format: str = Query(
        "markdown",
        description="Target format: 'markdown' (or 'md'), 'text' (or 'txt'), 'docx', 'pdf'"
    ),
    db: Session = Depends(get_db)
):
    """
    Exports lecture intelligence artifacts in requested format with clean layout,
    complete field preservation, and human-readable timestamps.
    """
    try:
        file_bytes, filename, media_type = ExportService.export_content(
            recording_id=recording_id,
            content_type=content_type,
            format=format,
            db=db
        )
        return Response(
            content=file_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except ExportException as e:
        if "not found" in e.message.lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export generation failed: {str(e)}"
        )
