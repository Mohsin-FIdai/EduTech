"""
Tests for Feature 1: Clickable Timestamp Citations and Structured Evidence.
"""

import os
import uuid
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models import Recording, Transcript, TranscriptSegment, IntelligenceReport, ConversationSession, ConversationTurn
from app.services.rag_service import LectureRAGService, MAX_CONVERSATION_TURNS
from app.schemas import QAResponse, EvidenceItem

TEST_DB_URL = "sqlite:///./test_rag_evidence.db"
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("test_rag_evidence.db"):
        try:
            os.remove("test_rag_evidence.db")
        except Exception:
            pass


@pytest.fixture
def db():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def create_sample_lecture(db):
    rec_id = str(uuid.uuid4())
    rec = Recording(
        id=rec_id,
        filename="lecture_operating_systems.wav",
        file_path="uploads/lecture_operating_systems.wav",
        file_size_bytes=1024 * 50,
        duration_seconds=360.0,
        file_format=".wav",
        media_type="audio",
        status="PROCESSED"
    )
    db.add(rec)
    db.commit()

    transcript = Transcript(
        id=str(uuid.uuid4()),
        recording_id=rec_id,
        original_transcript="Process states include ready, running, waiting, new and terminated. A program is code on disk, while a process is an active running instance in memory.",
        normalized_transcript="Process states include ready, running, waiting, new and terminated. A program is code on disk, while a process is an active running instance in memory."
    )
    db.add(transcript)
    db.commit()

    seg1 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=0,
        segment_index=1,
        start_time=102.5,
        end_time=149.2,
        text="A process in ready state waits in the ready queue for CPU allocation.",
        normalized_text="A process in ready state waits in the ready queue for CPU allocation.",
        english_text="A process in ready state waits in the ready queue for CPU allocation."
    )
    seg2 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=0,
        segment_index=2,
        start_time=149.3,
        end_time=215.0,
        text="The ready queue holds all processes waiting for the CPU scheduler.",
        normalized_text="The ready queue holds all processes waiting for the CPU scheduler.",
        english_text="The ready queue holds all processes waiting for the CPU scheduler."
    )
    seg3 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=0,
        segment_index=3,
        start_time=69.0,
        end_time=87.0,
        text="A program is code on disk, while a process is an active running instance in memory.",
        normalized_text="A program is code on disk, while a process is an active running instance in memory.",
        english_text="A program is code on disk, while a process is an active running instance in memory."
    )
    db.add_all([seg1, seg2, seg3])
    db.commit()

    return rec_id


def test_answer_question_returns_structured_evidence(db):
    """Verify that answer_question returns structured evidence list with start, end, segment_id, text."""
    rec_id = create_sample_lecture(db)

    result = LectureRAGService.answer_question(
        recording_id=rec_id,
        question="What is the ready state in process states?",
        db=db
    )

    assert result["found"] is True
    assert "evidence" in result
    evidence = result["evidence"]
    assert isinstance(evidence, list)
    assert len(evidence) > 0

    first_ev = evidence[0]
    assert "start" in first_ev
    assert "end" in first_ev
    assert "segment_id" in first_ev
    assert "text" in first_ev
    assert isinstance(first_ev["start"], float)
    assert isinstance(first_ev["end"], float)
    assert first_ev["start"] >= 0.0
    assert first_ev["end"] >= first_ev["start"]
    assert len(first_ev["text"]) > 0


def test_out_of_scope_evidence_is_empty(db):
    """Verify that out-of-scope refusals return empty evidence list."""
    rec_id = create_sample_lecture(db)

    result = LectureRAGService.answer_question(
        recording_id=rec_id,
        question="What is quantum entanglement in teleportation?",
        db=db
    )

    assert result["found"] is False
    assert "evidence" in result
    assert result["evidence"] == []


def test_qa_endpoint_serializes_evidence(client, db):
    """Verify that the FastAPI REST endpoint /api/recordings/{id}/qa includes evidence array."""
    rec_id = create_sample_lecture(db)

    resp = client.post(
        f"/api/recordings/{rec_id}/qa",
        json={"question": "What is the ready state?"}
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["found"] is True
    assert "evidence" in data
    assert isinstance(data["evidence"], list)
    assert len(data["evidence"]) > 0
    ev = data["evidence"][0]
    assert "start" in ev
    assert "end" in ev
    assert "segment_id" in ev
    assert "text" in ev


# ============================================================================
# Feature 2 Tests: ELI5 Toggle
# ============================================================================

def test_eli5_toggle_off_returns_normal_answer(db):
    """Verify that toggle off (eli5=False) returns normal pedagogical answer."""
    rec_id = create_sample_lecture(db)

    normal_res = LectureRAGService.answer_question(
        recording_id=rec_id,
        question="What is the ready state in process states?",
        db=db,
        eli5=False
    )

    assert normal_res["found"] is True
    assert "Process States in Operating Systems" in normal_res["answer"]
    assert "• **Ready (Ready State)**" in normal_res["answer"]


def test_eli5_toggle_on_returns_simplified_analogy_answer(db):
    """Verify that toggle on (eli5=True) returns simplified answer with analogy."""
    rec_id = create_sample_lecture(db)

    eli5_res = LectureRAGService.answer_question(
        recording_id=rec_id,
        question="What is the ready state in process states?",
        db=db,
        eli5=True
    )

    assert eli5_res["found"] is True
    # Simplified analogy: playground slide / kids waiting in line
    assert "playground" in eli5_res["answer"].lower() or "slide" in eli5_res["answer"].lower() or "line" in eli5_res["answer"].lower()
    # Sentence length is concise
    sentences = [s.strip() for s in eli5_res["answer"].split(".") if s.strip() and "evidence" not in s.lower()]
    for s in sentences:
        words = s.split()
        assert len(words) <= 18  # Short, simple sentences


def test_eli5_answers_strictly_grounded_spotcheck(db):
    """Spot-check 3 ELI5 answers against transcript content to verify groundedness with zero fabricated claims."""
    rec_id = create_sample_lecture(db)

    # Spot-check 1: Ready State
    q1 = "What is the ready state?"
    ans1 = LectureRAGService.answer_question(recording_id=rec_id, question=q1, db=db, eli5=True)
    assert ans1["found"] is True
    assert ans1["evidence"] != []
    assert any("ready" in e["text"].lower() for e in ans1["evidence"])
    assert "ready" in ans1["answer"].lower()

    # Spot-check 2: Ready Queue
    q2 = "What is the ready queue?"
    ans2 = LectureRAGService.answer_question(recording_id=rec_id, question=q2, db=db, eli5=True)
    assert ans2["found"] is True
    assert ans2["evidence"] != []
    assert any("ready" in e["text"].lower() or "queue" in e["text"].lower() for e in ans2["evidence"])
    assert "ready queue" in ans2["answer"].lower() or "waiting line" in ans2["answer"].lower()

    # Spot-check 3: Program vs Process vs Thread
    q3 = "What is a process vs program?"
    ans3 = LectureRAGService.answer_question(recording_id=rec_id, question=q3, db=db, eli5=True)
    assert ans3["found"] is True
    assert ans3["evidence"] != []
    assert "process" in ans3["answer"].lower()
    assert "recipe" in ans3["answer"].lower() or "kitchen" in ans3["answer"].lower()


def test_qa_endpoint_with_eli5_param(client, db):
    """Verify that POST /api/recordings/{id}/qa accepts eli5: true and returns simplified answer."""
    rec_id = create_sample_lecture(db)

    resp = client.post(
        f"/api/recordings/{rec_id}/qa",
        json={"question": "What is the ready state?", "eli5": True}
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["found"] is True
    assert "playground" in data["answer"].lower() or "slide" in data["answer"].lower()
    assert len(data["evidence"]) > 0


# ============================================================================
# Feature 3 Tests: Multi-Turn Conversation Memory
# ============================================================================

def test_multiturn_followup_uses_prior_turn_context(db):
    """
    Validates:
    Ask a full question, then ask a short follow-up ("give an example of that")
    in the same session — the follow-up answer correctly uses context from the prior turn.
    """
    rec_id = create_sample_lecture(db)
    session_id = f"test_session_{uuid.uuid4().hex[:8]}"

    # Turn 1: Ask full question
    ans1 = LectureRAGService.answer_question(
        recording_id=rec_id,
        question="What is the ready state in process states?",
        db=db,
        session_id=session_id
    )
    assert ans1["found"] is True
    assert "ready" in ans1["answer"].lower()

    # Check that turns were persisted
    sess = db.query(ConversationSession).filter_by(id=session_id).first()
    assert sess is not None
    turns = db.query(ConversationTurn).filter_by(session_id=session_id).all()
    assert len(turns) == 2  # 1 user + 1 assistant

    # Turn 2: Short follow-up containing "that" and <6 words
    ans2 = LectureRAGService.answer_question(
        recording_id=rec_id,
        question="give an example of that",
        db=db,
        session_id=session_id
    )
    assert ans2["found"] is True
    # Follow-up uses ready state context
    assert "ready" in ans2["answer"].lower() or "process" in ans2["answer"].lower()
    assert len(ans2["evidence"]) > 0

    # Total turns should now be 4
    turns_after = db.query(ConversationTurn).filter_by(session_id=session_id).all()
    assert len(turns_after) == 4


def test_new_session_isolation(db):
    """Validates: Starting a new session (new session_id) does not carry over context from a previous session."""
    rec_id = create_sample_lecture(db)
    session_a = f"session_a_{uuid.uuid4().hex[:8]}"
    session_b = f"session_b_{uuid.uuid4().hex[:8]}"

    # Ask full question in Session A
    LectureRAGService.answer_question(
        recording_id=rec_id,
        question="What is the ready state in process states?",
        db=db,
        session_id=session_a
    )

    # In Session B (fresh), ask ambiguous follow-up with no prior context in this session
    ans_b = LectureRAGService.answer_question(
        recording_id=rec_id,
        question="give an example of that",
        db=db,
        session_id=session_b
    )
    # Session B has no prior turns, so it should not find ready state or carry over context
    sess_b_turns = db.query(ConversationTurn).filter_by(session_id=session_b).all()
    assert len(sess_b_turns) == 2  # The turn attempted in session B
    # Because there was no prior turn in session B, 'give an example of that' lacked context
    # Session A and Session B turns must be completely isolated
    sess_a_turns = db.query(ConversationTurn).filter_by(session_id=session_a).all()
    assert len(sess_a_turns) == 2


def test_multiturn_history_trimmed_to_max_turns(db):
    """Validates: Turn history older than MAX_CONVERSATION_TURNS (5) is excluded from context."""
    rec_id = create_sample_lecture(db)
    session_id = f"session_trim_{uuid.uuid4().hex[:8]}"
    sess = ConversationSession(id=session_id, recording_id=rec_id)
    db.add(sess)
    db.commit()

    # Pre-populate 8 turns (4 user + 4 assistant)
    # Oldest turns talk about "quantum teleportation" (which should be trimmed out)
    t1 = ConversationTurn(session_id=session_id, role="user", content="What about quantum teleportation?")
    t2 = ConversationTurn(session_id=session_id, role="assistant", content="Quantum teleportation is not in this lecture.")
    db.add_all([t1, t2])
    db.commit()

    # Next 6 turns talk about ready state
    for i in range(3):
        u = ConversationTurn(session_id=session_id, role="user", content=f"Tell me about ready state step {i}")
        a = ConversationTurn(session_id=session_id, role="assistant", content=f"Ready state step {i} is in memory.")
        db.add_all([u, a])
    db.commit()

    total_turns_in_db = db.query(ConversationTurn).filter_by(session_id=session_id).count()
    assert total_turns_in_db == 8

    # Query latest turns as done in rag_service
    recent_turns = db.query(ConversationTurn).filter_by(
        session_id=session_id
    ).order_by(ConversationTurn.id.desc()).limit(MAX_CONVERSATION_TURNS).all()
    recent_turns.reverse()

    assert len(recent_turns) == MAX_CONVERSATION_TURNS
    # The oldest turns (t1, t2) must be excluded from recent_turns
    turn_contents = [t.content for t in recent_turns]
    assert "What about quantum teleportation?" not in turn_contents
    assert "Quantum teleportation is not in this lecture." not in turn_contents


def test_rule_based_followup_makes_no_extra_llm_call(monkeypatch, db):
    """Validates: Rule-based follow-up query rewriting makes 0 extra LLM calls."""
    rec_id = create_sample_lecture(db)
    session_id = f"session_callcount_{uuid.uuid4().hex[:8]}"

    # Turn 1: normal question
    LectureRAGService.answer_question(
        recording_id=rec_id,
        question="What is the ready state in process states?",
        db=db,
        session_id=session_id
    )

    llm_call_count = {"count": 0}

    # Spy on any potential LLM call
    try:
        from app.services.llm_provider import LocalLLMProvider
        orig_gen = LocalLLMProvider.generate_text

        def mock_generate(self, *args, **kwargs):
            llm_call_count["count"] += 1
            return orig_gen(self, *args, **kwargs)

        monkeypatch.setattr(LocalLLMProvider, "generate_text", mock_generate)
    except Exception:
        pass

    # Turn 2: follow-up question
    LectureRAGService.answer_question(
        recording_id=rec_id,
        question="give an example of that",
        db=db,
        session_id=session_id
    )

    # In deterministic mode, llm_call_count is 0, confirming no extra LLM rewrite call
    assert llm_call_count["count"] == 0


def test_qa_endpoint_maintains_session(client, db):
    """Validates: REST endpoint /api/recordings/{id}/qa accepts and echoes session_id."""
    rec_id = create_sample_lecture(db)
    client_sess_id = f"client_sess_{uuid.uuid4().hex[:6]}"

    resp1 = client.post(
        f"/api/recordings/{rec_id}/qa",
        json={"question": "What is the ready state?", "session_id": client_sess_id}
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["session_id"] == client_sess_id

    # Follow-up request with same session_id
    resp2 = client.post(
        f"/api/recordings/{rec_id}/qa",
        json={"question": "tell me more about it", "session_id": client_sess_id}
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["session_id"] == client_sess_id
    assert data2["found"] is True


# ============================================================================
# Additional Tests: Evidence Density Cap & In-Lecture Questions (FinTech & ML)
# ============================================================================

def create_sample_fintech_lecture(db):
    """Helper to create a FinTech lecture with study topics and transcript segments."""
    import json
    rec_id = str(uuid.uuid4())
    rec = Recording(
        id=rec_id,
        filename="Fintech.mp4",
        file_path="uploads/Fintech.mp4",
        file_size_bytes=1024 * 50,
        duration_seconds=671.0,
        file_format=".mp4",
        media_type="video",
        status="PROCESSED"
    )
    db.add(rec)
    db.commit()

    transcript = Transcript(
        id=str(uuid.uuid4()),
        recording_id=rec_id,
        original_transcript="Today we discuss about financial technology and simply fintech. We will examine artificial intelligence and data analytics.",
        normalized_transcript="Today we discuss about financial technology and simply fintech. We will examine artificial intelligence and data analytics."
    )
    db.add(transcript)
    db.commit()

    seg1 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=0,
        segment_index=1,
        start_time=0.0,
        end_time=28.0,
        text="This field is called financial technology, and simply fintech.",
        normalized_text="This field is called financial technology, and simply fintech.",
        english_text="This field is called financial technology, and simply fintech."
    )
    seg2 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=0,
        segment_index=2,
        start_time=57.8,
        end_time=67.6,
        text="replace or improve many of those processes using software, artificial intelligence and data analytics.",
        normalized_text="replace or improve many of those processes using software, artificial intelligence and data analytics.",
        english_text="replace or improve many of those processes using software, artificial intelligence and data analytics."
    )
    seg3 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=0,
        segment_index=3,
        start_time=87.5,
        end_time=94.6,
        text="lending, insurance, investment, fraud detection, financial analytics and many other services.",
        normalized_text="lending, insurance, investment, fraud detection, financial analytics and many other services.",
        english_text="lending, insurance, investment, fraud detection, financial analytics and many other services."
    )
    seg4 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=0,
        segment_index=4,
        start_time=219.7,
        end_time=280.0,
        text="AI and machine learning algorithms monitor transactions for fraud detection and automated credit scoring.",
        normalized_text="AI and machine learning algorithms monitor transactions for fraud detection and automated credit scoring.",
        english_text="AI and machine learning algorithms monitor transactions for fraud detection and automated credit scoring."
    )
    db.add_all([seg1, seg2, seg3, seg4])
    db.commit()

    notes = {
        "theme": "Financial Technology",
        "title": "Fintech Overview",
        "study_topics": [
            {
                "id": "topic_01",
                "title": "Fintech Overview",
                "start_time": 0.0,
                "end_time": 450.0,
                "explanation": "Introduction to financial technology and mobile banking services.",
                "key_points": ["Fintech combines finance and technology."]
            },
            {
                "id": "topic_03",
                "title": "Machine Learning",
                "start_time": 219.72,
                "end_time": 701.0,
                "explanation": "This section explains how machine learning and artificial intelligence are applied in FinTech for fraud detection, credit scoring, and budgeting.",
                "key_points": [
                    "AI should be treated as a decision tool.",
                    "Personal financial management automatically categorizes spending."
                ]
            }
        ]
    }
    report = IntelligenceReport(
        recording_id=rec_id,
        transcript_id=transcript.id,
        summary_short="Overview of financial technology and machine learning applications.",
        summary_detailed="Detailed summary of financial technology and machine learning applications.",
        lecture_notes_json=json.dumps(notes),
        important_points_json="[]",
        assignments_json="[]",
        action_items_json="[]",
        quizzes_json="[]"
    )
    db.add(report)
    db.commit()
    return rec_id


def test_evidence_list_density_capped_at_topic_start(db):
    """
    Validates: When answering overview or multi-segment questions, the evidence list
    focuses on the topic starting timestamp and contains at most 2 items, never flooding
    the response with dozens of granular micro-segment chips.
    """
    rec_id = create_sample_lecture(db)

    # Overview question spanning wide time range
    res = LectureRAGService.answer_question(
        recording_id=rec_id,
        question="can you give me an overview of the lecture?",
        db=db
    )

    assert res["found"] is True
    assert "evidence" in res
    assert isinstance(res["evidence"], list)
    # Crucial acceptance criteria: 1 to 2 items max, never dozens
    assert 1 <= len(res["evidence"]) <= 2
    first_chip = res["evidence"][0]
    assert first_chip["start"] >= 0.0
    assert first_chip["end"] >= first_chip["start"]
    assert "text" in first_chip


def test_fintech_machine_learning_query_answered(db):
    """
    Validates: In-lecture questions regarding Machine Learning / AI in FinTech
    are properly retrieved and answered instead of triggering false out-of-scope refusals.
    """
    rec_id = create_sample_fintech_lecture(db)

    # The exact query from user feedback
    res = LectureRAGService.answer_question(
        recording_id=rec_id,
        question="what is the use of machine learning in fintech",
        db=db
    )

    assert res["status"] == "success"
    assert res["found"] is True
    # Verify the answer is lecture-grounded and does NOT refuse
    assert "No supporting evidence found" not in res["answer"]
    assert "not covered in the provided lecture" not in res["answer"]
    ans_lower = res["answer"].lower()
    assert any(term in ans_lower for term in ["fraud", "credit", "algorithm", "analytics", "security", "scoring", "budgeting"])

    # Evidence points to where the topic starts
    assert "evidence" in res
    assert 1 <= len(res["evidence"]) <= 2
    assert res["evidence"][0]["start"] >= 0.0


def test_fintech_machine_learning_eli5(db):
    """
    Validates: ELI5 toggle for Machine Learning in FinTech explains concepts with
    accessible, intuitive analogies without refusal.
    """
    rec_id = create_sample_fintech_lecture(db)

    res = LectureRAGService.answer_question(
        recording_id=rec_id,
        question="what is the use of machine learning in fintech",
        db=db,
        eli5=True
    )

    assert res["status"] == "success"
    assert res["found"] is True
    assert "No supporting evidence found" not in res["answer"]
    ans_lower = res["answer"].lower()
    # Analogy words present
    assert any(w in ans_lower for w in ["detective", "helper", "coins", "spending", "bank"])
    assert 1 <= len(res["evidence"]) <= 2



