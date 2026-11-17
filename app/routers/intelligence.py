"""
intelligence.py — REST API router for Lecture & Meeting Intelligence layer.
"""

import json
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.database import get_db, SessionLocal
from app.models import Recording, IntelligenceReport, Transcript, TranscriptSegment
from app.schemas import (
    IntelligenceResponse,
    QARequest,
    QAResponse,
    LectureNotes,
    AssignmentItem,
    ActionItem,
    QuizQuestion
)
from app.services.intelligence_service import LectureIntelligenceService, IntelligenceException

router = APIRouter(prefix="/api/recordings", tags=["Lecture & Meeting Intelligence"])


@router.post(
    "/{recording_id}/intelligence/generate",
    response_model=IntelligenceResponse,
    summary="Generate and persist full lecture & meeting intelligence report"
)
def generate_intelligence_endpoint(
    recording_id: str,
    db: Session = Depends(get_db)
):
    try:
        data = LectureIntelligenceService.generate_intelligence_report(recording_id, db)
        return IntelligenceResponse(
            status="success",
            recording_id=data["recording_id"],
            transcript_id=data["transcript_id"],
            summary_short=data["summary_short"],
            summary_detailed=data["summary_detailed"],
            lecture_notes=LectureNotes(**data["lecture_notes"]),
            assignments=[AssignmentItem(**a) for a in data["assignments"]],
            action_items=[ActionItem(**ai) for ai in data["action_items"]],
            quizzes=[QuizQuestion(**q) for q in data["quizzes"]],
            created_at=data["created_at"]
        )
    except IntelligenceException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Intelligence generation failed: {str(e)}"
        )


@router.get(
    "/{recording_id}/intelligence",
    response_model=IntelligenceResponse,
    summary="Retrieve existing lecture & meeting intelligence report"
)
def get_intelligence_endpoint(
    recording_id: str,
    db: Session = Depends(get_db)
):
    report = db.query(IntelligenceReport).filter(IntelligenceReport.recording_id == recording_id).first()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Intelligence report for recording '{recording_id}' not found. Please generate first."
        )

    try:
        notes_data = json.loads(report.lecture_notes_json)
        if notes_data.get("universal_notes_rev") != "2026-11-10-v3":
            try:
                regen = LectureIntelligenceService.generate_intelligence_report(recording_id, db, force=True)
                notes_data = regen["lecture_notes"]
                assignments_data = regen["assignments"]
                action_items_data = regen["action_items"]
                quizzes_data = regen["quizzes"]
            except Exception:
                assignments_data = json.loads(report.assignments_json)
                action_items_data = json.loads(report.action_items_json)
                quizzes_data = json.loads(report.quizzes_json)
        else:
            assignments_data = json.loads(report.assignments_json)
            action_items_data = json.loads(report.action_items_json)
            quizzes_data = json.loads(report.quizzes_json)

        return IntelligenceResponse(
            status="success",
            recording_id=report.recording_id,
            transcript_id=report.transcript_id,
            summary_short=report.summary_short,
            summary_detailed=report.summary_detailed,
            lecture_notes=LectureNotes(**notes_data),
            assignments=[AssignmentItem(**a) for a in assignments_data],
            action_items=[ActionItem(**ai) for ai in action_items_data],
            quizzes=[QuizQuestion(**q) for q in quizzes_data],
            created_at=report.created_at
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load intelligence report: {str(e)}"
        )


@router.post(
    "/{recording_id}/qa",
    response_model=QAResponse,
    summary="Grounded Q&A strictly answered from recording transcripts with source timestamps"
)
@router.post(
    "/{recording_id}/intelligence/qa",
    response_model=QAResponse,
    summary="Grounded Q&A strictly answered from recording transcripts with source timestamps"
)
def qa_endpoint(
    recording_id: str,
    request: QARequest,
    db: Session = Depends(get_db)
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording '{recording_id}' not found."
        )

    try:
        result = LectureIntelligenceService.answer_grounded_question(
            request.question,
            recording_id,
            db,
            eli5=bool(request.eli5),
            session_id=request.session_id
        )
        return QAResponse(
            status="success",
            found=result["found"],
            question=result["question"],
            answer=result["answer"],
            timestamp=result["timestamp"],
            start_time=result.get("start_time"),
            end_time=result.get("end_time"),
            source_segment=result.get("source_segment"),
            confidence=result.get("confidence", 0.0),
            citations=result.get("citations", []),
            evidence=result.get("evidence", []),
            session_id=result.get("session_id")
        )
    except IntelligenceException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grounded Q&A failed: {str(e)}"
        )


@router.post(
    "/{recording_id}/qa/stream",
    summary="Stream Q&A response token-by-token over Server-Sent Events"
)
@router.post(
    "/{recording_id}/intelligence/qa/stream",
    summary="Stream Q&A response token-by-token over Server-Sent Events"
)
async def qa_stream_endpoint(
    recording_id: str,
    request: QARequest,
    db: Session = Depends(get_db)
):
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording '{recording_id}' not found."
        )

    import asyncio
    import threading
    from app.services.rag_service import LectureRAGService

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()
    cancel_flag = threading.Event()

    def _worker():
        # Run entire generator + DB session on ONE dedicated OS thread
        worker_db = SessionLocal()
        try:
            gen = LectureRAGService.stream_answer_question(
                recording_id=recording_id,
                question=request.question,
                db=worker_db,
                eli5=bool(request.eli5),
                session_id=request.session_id,
            )
            try:
                for event in gen:
                    if cancel_flag.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, f"data: {json.dumps(event, ensure_ascii=False)}\n\n")
            finally:
                gen.close()
        except Exception as exc:
            if not cancel_flag.is_set():
                loop.call_soon_threadsafe(queue.put_nowait, f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n")
        finally:
            try:
                worker_db.close()
            except Exception:
                pass
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    threading.Thread(target=_worker, daemon=True).start()

    async def _event_stream():
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                yield item
        finally:
            cancel_flag.set()

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/{recording_id}/intelligence/quizzes",
    summary="Retrieve practice quizzes for a recording"
)
def get_quizzes_endpoint(
    recording_id: str,
    db: Session = Depends(get_db)
):
    report = db.query(IntelligenceReport).filter(IntelligenceReport.recording_id == recording_id).first()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Intelligence report for recording '{recording_id}' not found. Please generate first."
        )

    try:
        quizzes = json.loads(report.quizzes_json)
        return {
            "status": "success",
            "recording_id": recording_id,
            "total_quizzes": len(quizzes),
            "quizzes": quizzes
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch quizzes: {str(e)}"
        )


def _run_quiz_regeneration_task(recording_id: str) -> None:
    """Background worker that regenerates only quizzes and saves progress incrementally."""
    db = SessionLocal()
    try:
        report = db.query(IntelligenceReport).filter(IntelligenceReport.recording_id == recording_id).first()
        if not report:
            return
        recording = db.query(Recording).filter(Recording.id == recording_id).first()
        transcript = db.query(Transcript).filter(Transcript.recording_id == recording_id).first()
        segments = []
        if transcript:
            segments = (
                db.query(TranscriptSegment)
                .filter(TranscriptSegment.transcript_id == transcript.id)
                .order_by(TranscriptSegment.segment_index.asc())
                .all()
            )
        if not segments:
            return

        lecture_notes = json.loads(report.lecture_notes_json or "{}")
        assignments = json.loads(report.assignments_json or "[]")
        text_corpus = " ".join(
            (getattr(s, "english_text", None) or s.normalized_text or s.text or "")
            for s in segments
        ).strip()
        topic_hint = (
            (lecture_notes.get("study_topics") or [{}])[0].get("title")
            or (recording.filename if recording else None)
        )

        def _on_progress(partial_quizzes: list[dict], target_count: int) -> None:
            try:
                lecture_notes.setdefault("meta", {})["quiz_generation"] = {
                    "status": "running",
                    "generated": len(partial_quizzes),
                    "target": target_count,
                }
                report.quizzes_json = json.dumps(partial_quizzes)
                report.lecture_notes_json = json.dumps(lecture_notes)
                db.commit()
            except Exception:
                db.rollback()

        from app.services.study_guide_generator import StudyGuideSynthesizer
        quiz_status: dict = {}
        quizzes = StudyGuideSynthesizer.generate_conceptual_quizzes(
            transcript_text=text_corpus,
            topic_hint=topic_hint,
            lecture_notes=lecture_notes,
            assignments=assignments,
            segments=segments,
            allow_extractive_fallback=False,
            status_out=quiz_status,
            on_progress=_on_progress,
        )
        lecture_notes.setdefault("meta", {})["quiz_generation"] = quiz_status
        report.quizzes_json = json.dumps(quizzes)
        report.lecture_notes_json = json.dumps(lecture_notes)
        db.commit()
    except Exception as e:
        db.rollback()
        try:
            report = db.query(IntelligenceReport).filter(IntelligenceReport.recording_id == recording_id).first()
            if report:
                notes = json.loads(report.lecture_notes_json or "{}")
                notes.setdefault("meta", {})["quiz_generation"] = {"status": "failed", "reason": str(e)}
                report.lecture_notes_json = json.dumps(notes)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


@router.post(
    "/{recording_id}/intelligence/quizzes/regenerate",
    summary="Regenerate ONLY the practice quizzes for an existing intelligence report"
)
def regenerate_quizzes_only_endpoint(
    recording_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    report = db.query(IntelligenceReport).filter(IntelligenceReport.recording_id == recording_id).first()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Intelligence report for recording '{recording_id}' not found. Please generate first."
        )

    transcript = db.query(Transcript).filter(Transcript.recording_id == recording_id).first()
    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No transcript found for this recording."
        )

    lecture_notes = json.loads(report.lecture_notes_json or "{}")
    lecture_notes.setdefault("meta", {})["quiz_generation"] = {
        "status": "running",
        "generated": 0,
        "target": 10,
    }
    report.lecture_notes_json = json.dumps(lecture_notes)
    db.commit()

    background_tasks.add_task(_run_quiz_regeneration_task, recording_id)
    return {
        "status": "started",
        "recording_id": recording_id,
        "quiz_generation": lecture_notes["meta"]["quiz_generation"],
    }
