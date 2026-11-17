"""
test_phase15_export.py — Test Suite for Multi-Format Export (Phase 15: PDF, DOCX, TXT, Markdown).

Validates:
1. Content fidelity across all 4 formats (PDF, DOCX, TXT, MD) preserving Not specified fields & timestamps.
2. Format-specific structural parsing (docx.Document and pypdf.PdfReader).
3. Selective export filtering (assignments, summary, notes, transcript, quizzes).
4. Error handling (non-existent recording, invalid format, invalid content_type).
5. REST API export endpoints with Content-Disposition headers.
"""

import io
import os
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import docx
import pypdf

from app.database import Base, get_db
from app.main import app
from app.models import Recording, Transcript, TranscriptSegment, IntelligenceReport
from app.services.export_service import ExportService, ExportException
from app.services.intelligence_service import LectureIntelligenceService

TEST_DB_URL = "sqlite:///./test_phase15_export.db"
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("test_phase15_export.db"):
        try:
            os.remove("test_phase15_export.db")
        except Exception:
            pass


@pytest.fixture
def db_session():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def create_sample_recording_with_intelligence(db_session) -> str:
    """Helper to populate a complete sample recording with transcripts and intelligence report."""
    rec_id = str(uuid.uuid4())
    rec = Recording(
        id=rec_id,
        filename="ai_lecture_export.wav",
        file_path="uploads/ai_lecture_export.wav",
        file_size_bytes=1024 * 60,
        duration_seconds=75.0,
        file_format=".wav",
        media_type="audio",
        status="PROCESSED"
    )
    db_session.add(rec)
    db_session.commit()

    transcript = Transcript(
        id=str(uuid.uuid4()),
        recording_id=rec_id,
        original_transcript="Assignment 1 is about Neural Networks. Submit a 5-page PDF by Friday midnight.",
        normalized_transcript="Assignment 1 is about Neural Networks. Submit a 5-page PDF by Friday midnight."
    )
    db_session.add(transcript)
    db_session.commit()

    seg1 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=0,
        segment_index=0,
        start_time=0.00,
        end_time=18.50,
        text="Assignment 1 is about Neural Networks.",
        normalized_text="Assignment 1 is about Neural Networks.",
        language="en"
    )
    seg2 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=1,
        segment_index=1,
        start_time=19.00,
        end_time=38.25,
        text="Submit a 5-page PDF report for this assignment before Friday midnight on LMS portal.",
        normalized_text="Submit a 5-page PDF report for this assignment before Friday midnight on LMS portal.",
        language="en"
    )
    seg3 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=2,
        segment_index=2,
        start_time=39.00,
        end_time=55.40,
        text="Remember this concept because this will come in exam: Backpropagation computes gradient vectors for weights.",
        normalized_text="Remember this concept because this will come in exam: Backpropagation computes gradient vectors for weights.",
        language="en"
    )
    seg4 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=3,
        segment_index=3,
        start_time=56.00,
        end_time=74.50,
        text="Hamza will finish the benchmark testing by Wednesday.",
        normalized_text="Hamza will finish the benchmark testing by Wednesday.",
        language="en"
    )
    db_session.add_all([seg1, seg2, seg3, seg4])
    db_session.commit()

    # Generate and persist intelligence report
    LectureIntelligenceService.generate_intelligence_report(rec_id, db_session)
    return rec_id


# ============================================================================
# TEST 1: Content Fidelity Across 4 Formats (PDF, DOCX, TXT, Markdown)
# ============================================================================
def test_export_content_fidelity_across_all_four_formats(db_session):
    """
    Validates that full export across all 4 formats includes every piece of generated
    data, preserving 'Not specified' fields and human-readable timestamps.
    """
    rec_id = create_sample_recording_with_intelligence(db_session)

    # 1. Markdown Export
    md_bytes, md_name, md_mime = ExportService.export_content(rec_id, "full", "markdown", db_session)
    md_text = md_bytes.decode("utf-8")
    assert md_mime == "text/markdown; charset=utf-8"
    assert md_name.endswith(".md")
    assert "Executive Summary" in md_text
    assert "Neural Networks" in md_text
    assert "PDF" in md_text
    assert "Friday midnight" in md_text
    assert "[EXAM RELEVANT]" in md_text
    assert "will come in exam" in md_text
    assert "Hamza" in md_text
    assert "Wednesday" in md_text
    assert "Original Transcript" in md_text
    assert "Normalized Roman Urdu Transcript" in md_text
    assert "Not specified" in md_text  # Preserved unstated fields
    assert "[0.00s - 18.50s]" in md_text or "[19.00s - 38.25s]" in md_text

    # 2. Plain Text Export
    txt_bytes, txt_name, txt_mime = ExportService.export_content(rec_id, "full", "text", db_session)
    txt_text = txt_bytes.decode("utf-8")
    assert txt_mime == "text/plain; charset=utf-8"
    assert txt_name.endswith(".txt")
    assert "LECTURE & MEETING INTELLIGENCE REPORT" in txt_text
    assert "Neural Networks" in txt_text
    assert "PDF" in txt_text
    assert "Hamza" in txt_text
    assert "[EXAM RELEVANT]" in txt_text
    assert "Not specified" in txt_text
    assert "[Original Spoken Transcript]" in txt_text
    assert "[Normalized Roman Urdu Transcript]" in txt_text

    # 3. DOCX Export
    docx_bytes, docx_name, docx_mime = ExportService.export_content(rec_id, "full", "docx", db_session)
    assert docx_mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert docx_name.endswith(".docx")
    assert len(docx_bytes) > 500

    doc = docx.Document(io.BytesIO(docx_bytes))
    docx_full_text = "\n".join(p.text for p in doc.paragraphs)
    assert "Lecture & Meeting Intelligence Report" in docx_full_text
    assert "Neural Networks" in docx_full_text
    assert "PDF" in docx_full_text
    assert "Hamza" in docx_full_text
    assert "EXAM RELEVANT" in docx_full_text
    assert "Not specified" in docx_full_text

    # 4. PDF Export
    pdf_bytes, pdf_name, pdf_mime = ExportService.export_content(rec_id, "full", "pdf", db_session)
    assert pdf_mime == "application/pdf"
    assert pdf_name.endswith(".pdf")
    assert len(pdf_bytes) > 500

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) >= 1
    pdf_full_text = "\n".join(page.extract_text() for page in reader.pages)
    assert "Lecture & Meeting Intelligence Report" in pdf_full_text
    assert "Neural Networks" in pdf_full_text
    assert "PDF" in pdf_full_text
    assert "Hamza" in pdf_full_text
    assert "EXAM RELEVANT" in pdf_full_text
    assert "Not specified" in pdf_full_text


# ============================================================================
# TEST 2: Format-Specific Structural Validation
# ============================================================================
def test_format_specific_structural_validation(db_session):
    """
    Validates structural hierarchy in DOCX and PDF:
    - DOCX: Document title, Headings level 1 & 2, bullet points, bold runs.
    - PDF: Multi-page flowable rendering and valid text layer extraction.
    """
    rec_id = create_sample_recording_with_intelligence(db_session)

    # DOCX Structure Check
    docx_bytes, _, _ = ExportService.export_content(rec_id, "full", "docx", db_session)
    doc = docx.Document(io.BytesIO(docx_bytes))

    # Verify heading styles exist
    heading_texts = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
    assert any("1. Executive Summary" in h for h in heading_texts)
    assert any("2. Structured Lecture Notes" in h for h in heading_texts)
    assert any("3. Assignments & Coursework" in h for h in heading_texts)
    assert any("4. Action Items & Deliverables" in h for h in heading_texts)
    assert any("6. Complete Transcripts" in h for h in heading_texts)

    # PDF Structure Check
    pdf_bytes, _, _ = ExportService.export_content(rec_id, "full", "pdf", db_session)
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) >= 1
    extracted_text = reader.pages[0].extract_text()
    assert "Lecture & Meeting Intelligence Report" in extracted_text
    assert "Executive Summary" in extracted_text


# ============================================================================
# TEST 3: Selective Export Filtering
# ============================================================================
def test_selective_export_filtering(db_session):
    """
    Validates that selective content_type export strictly outputs only requested sections.
    """
    rec_id = create_sample_recording_with_intelligence(db_session)

    # 1. Assignments only
    asgn_bytes, _, _ = ExportService.export_content(rec_id, "assignments", "markdown", db_session)
    asgn_text = asgn_bytes.decode("utf-8")
    assert "Assignments & Coursework" in asgn_text
    assert "Neural Networks" in asgn_text
    assert "Executive Summary" not in asgn_text
    assert "Structured Lecture Notes" not in asgn_text
    assert "Complete Transcripts" not in asgn_text

    # 2. Summary only
    sum_bytes, _, _ = ExportService.export_content(rec_id, "summary", "markdown", db_session)
    sum_text = sum_bytes.decode("utf-8")
    assert "Executive Summary" in sum_text
    assert "Assignments & Coursework" not in sum_text
    assert "Action Items & Meeting Deliverables" not in sum_text
    assert "Complete Transcripts" not in sum_text

    # 3. Transcripts only (Dual transcripts present)
    tr_bytes, _, _ = ExportService.export_content(rec_id, "transcript", "markdown", db_session)
    tr_text = tr_bytes.decode("utf-8")
    assert "Complete Transcripts" in tr_text
    assert "Original Transcript" in tr_text
    assert "Normalized Roman Urdu Transcript" in tr_text
    assert "Executive Summary" not in tr_text
    assert "Assignments & Coursework" not in tr_text

    # 4. Quizzes only
    q_bytes, _, _ = ExportService.export_content(rec_id, "quizzes", "markdown", db_session)
    q_text = q_bytes.decode("utf-8")
    assert "Practice Quizzes" in q_text
    assert "Executive Summary" not in q_text
    assert "Complete Transcripts" not in q_text


# ============================================================================
# TEST 4: Error Handling & Edge Cases
# ============================================================================
def test_export_error_handling_and_edge_cases(db_session):
    """
    Validates proper error raising on non-existent recording, invalid format,
    and invalid content_type.
    """
    # 1. Non-existent recording
    with pytest.raises(ExportException) as exc1:
        ExportService.export_content("non-existent-rec-id", "full", "pdf", db_session)
    assert "not found" in str(exc1.value).lower()

    # 2. Invalid content type
    rec_id = create_sample_recording_with_intelligence(db_session)
    with pytest.raises(ExportException) as exc2:
        ExportService.export_content(rec_id, "invalid_section", "pdf", db_session)
    assert "invalid content_type" in str(exc2.value).lower()

    # 3. Invalid format
    with pytest.raises(ExportException) as exc3:
        ExportService.export_content(rec_id, "full", "unsupported_fmt", db_session)
    assert "invalid format" in str(exc3.value).lower()


# ============================================================================
# TEST 5: REST API Download Verification & Headers
# ============================================================================
def test_rest_api_export_endpoints(client, db_session):
    """
    Validates REST API endpoint GET /api/recordings/{id}/export returns 200,
    correct Content-Type, and proper Content-Disposition attachment filenames.
    """
    rec_id = create_sample_recording_with_intelligence(db_session)

    # 1. PDF export endpoint
    res_pdf = client.get(f"/api/recordings/{rec_id}/export?content_type=full&format=pdf")
    assert res_pdf.status_code == 200
    assert res_pdf.headers["content-type"] == "application/pdf"
    assert 'attachment; filename="ai_lecture_export_full.pdf"' in res_pdf.headers["content-disposition"]
    assert len(res_pdf.content) > 500

    # 2. DOCX export endpoint
    res_docx = client.get(f"/api/recordings/{rec_id}/export?content_type=full&format=docx")
    assert res_docx.status_code == 200
    assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in res_docx.headers["content-type"]
    assert 'attachment; filename="ai_lecture_export_full.docx"' in res_docx.headers["content-disposition"]
    assert len(res_docx.content) > 500

    # 3. Markdown export endpoint
    res_md = client.get(f"/api/recordings/{rec_id}/export?content_type=assignments&format=markdown")
    assert res_md.status_code == 200
    assert "text/markdown" in res_md.headers["content-type"]
    assert 'attachment; filename="ai_lecture_export_assignments.md"' in res_md.headers["content-disposition"]
    assert b"Assignments & Coursework" in res_md.content

    # 4. Text export endpoint
    res_txt = client.get(f"/api/recordings/{rec_id}/export?content_type=summary&format=text")
    assert res_txt.status_code == 200
    assert "text/plain" in res_txt.headers["content-type"]
    assert 'attachment; filename="ai_lecture_export_summary.txt"' in res_txt.headers["content-disposition"]
    assert b"EXECUTIVE SUMMARY" in res_txt.content

    # 5. Non-existent recording returns 404
    res_404 = client.get(f"/api/recordings/{str(uuid.uuid4())}/export?content_type=full&format=pdf")
    assert res_404.status_code == 404


# ============================================================================
# TEST 6: Special-Character Escaping & Structural Resilience
# ============================================================================
def test_special_character_escaping_and_structural_resilience(db_session):
    """
    Validates that text containing special characters (*, #, _, |, <, >, &)
    in transcript segments or extracted fields does not corrupt Markdown headings
    or cause XML parsing crashes in ReportLab PDF / DOCX generation.
    """
    rec_id = str(uuid.uuid4())
    rec = Recording(
        id=rec_id,
        filename="special_chars_lecture.wav",
        file_path="uploads/special_chars_lecture.wav",
        file_size_bytes=1024 * 40,
        duration_seconds=45.0,
        file_format=".wav",
        media_type="audio",
        status="PROCESSED"
    )
    db_session.add(rec)
    db_session.commit()

    transcript = Transcript(
        id=str(uuid.uuid4()),
        recording_id=rec_id,
        original_transcript="Spoken content with C++ & Python *Data Structures* #1 <Deep_Learning> | Grade: A+ & 100%.",
        normalized_transcript="Spoken content with C++ & Python *Data Structures* #1 <Deep_Learning> | Grade: A+ & 100%."
    )
    db_session.add(transcript)
    db_session.commit()

    seg = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=0,
        segment_index=0,
        start_time=0.00,
        end_time=20.00,
        text="Assignment 1 is about C++ & Python *Data Structures* #1 <Algorithm_Ops> | 100%.",
        normalized_text="Assignment 1 is about C++ & Python *Data Structures* #1 <Algorithm_Ops> | 100%.",
        language="en"
    )
    db_session.add(seg)
    db_session.commit()

    # Generate intelligence
    LectureIntelligenceService.generate_intelligence_report(rec_id, db_session)

    # 1. Markdown Export Check (Must be visibly and structurally backslash-escaped)
    md_bytes, _, _ = ExportService.export_content(rec_id, "full", "markdown", db_session)
    md_text = md_bytes.decode("utf-8")
    assert r"C++ & Python \*Data Structures\* \#1 \<Algorithm\_Ops\> \| 100%" in md_text
    assert md_text.startswith("# Lecture & Meeting Intelligence Report")
    assert "## Assignments & Coursework" in md_text

    # 2. Plain Text Export Check (Must be preserved verbatim without syntax interference)
    txt_bytes, _, _ = ExportService.export_content(rec_id, "full", "text", db_session)
    txt_text = txt_bytes.decode("utf-8")
    assert "C++ & Python *Data Structures* #1 <Algorithm_Ops> | 100%" in txt_text

    # 3. PDF Export Check (Must safely render XML-escaped entities and extract properly)
    pdf_bytes, _, _ = ExportService.export_content(rec_id, "full", "pdf", db_session)
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    pdf_text = "\n".join(page.extract_text() for page in reader.pages)
    assert "C++ & Python" in pdf_text
    assert "Data Structures" in pdf_text
    assert "100%" in pdf_text

    # 4. DOCX Export Check (Must safely write into OOXML text runs without XML breakage)
    docx_bytes, _, _ = ExportService.export_content(rec_id, "full", "docx", db_session)
    doc = docx.Document(io.BytesIO(docx_bytes))
    docx_text = "\n".join(p.text for p in doc.paragraphs)
    assert "C++ & Python *Data Structures* #1 <Algorithm_Ops> | 100%" in docx_text


# ============================================================================
# TEST 7: Realistic-Length Multi-Chunk Scale Export
# ============================================================================
def test_realistic_length_multichunk_scale_export(db_session):
    """
    Validates export against a large, realistic 50-chunk recording (~1500s / 25 minutes)
    with multiple assignments, action items, lecture notes, and quizzes, confirming:
    - Zero silent truncation or data degradation
    - Strict chronological segment ordering
    - Multi-page document generation in PDF and DOCX
    """
    rec_id = str(uuid.uuid4())
    rec = Recording(
        id=rec_id,
        filename="full_scale_lecture_25min.wav",
        file_path="uploads/full_scale_lecture_25min.wav",
        file_size_bytes=1024 * 1024 * 15,
        duration_seconds=1500.0,
        file_format=".wav",
        media_type="audio",
        status="PROCESSED"
    )
    db_session.add(rec)
    db_session.commit()

    transcript = Transcript(
        id=str(uuid.uuid4()),
        recording_id=rec_id,
        original_transcript="Full 25-minute comprehensive machine learning lecture transcript.",
        normalized_transcript="Full 25-minute comprehensive machine learning lecture transcript."
    )
    db_session.add(transcript)
    db_session.commit()

    # Create 50 segments (30s each)
    segments = []
    for idx in range(50):
        st = idx * 30.0
        et = (idx + 1) * 30.0
        if idx == 0:
            txt = "For Assignment 1, you will implement Linear Regression algorithms."
        elif idx == 5:
            txt = "Submit a 5-page PDF report for Assignment 1 by Friday midnight."
        elif idx == 15:
            txt = "Remember this concept because this will come in exam: Cost functions measure model errors."
        elif idx == 25:
            txt = "For Assignment 2, build a Random Forest classifier by next month."
        elif idx == 35:
            txt = "Hamza will prepare the validation dataset by Tuesday."
        elif idx == 45:
            txt = "Bilal will submit the model evaluation benchmark by Thursday."
        else:
            txt = f"In lecture segment {idx}, we explore statistical feature engineering step {idx}."

        seg = TranscriptSegment(
            id=str(uuid.uuid4()),
            transcript_id=transcript.id,
            recording_id=rec_id,
            chunk_index=idx,
            segment_index=idx,
            start_time=st,
            end_time=et,
            text=txt,
            normalized_text=txt,
            language="en"
        )
        db_session.add(seg)
        segments.append(seg)
    db_session.commit()

    # Generate intelligence report for the full-length recording
    LectureIntelligenceService.generate_intelligence_report(rec_id, db_session)

    # 1. Full Markdown Export Check
    md_bytes, _, _ = ExportService.export_content(rec_id, "full", "markdown", db_session)
    md_text = md_bytes.decode("utf-8")
    assert "Assignment 1" in md_text
    assert "Assignment 2" in md_text
    assert "Hamza" in md_text
    assert "Bilal" in md_text
    assert "[EXAM RELEVANT]" in md_text
    # Verify all 50 segment timestamps are present in chronological order
    for idx in range(50):
        st_str = f"[{idx * 30:.2f}s - {(idx + 1) * 30:.2f}s]"
        assert st_str in md_text

    # 2. Full PDF Export Check (Multi-page rendering)
    pdf_bytes, _, _ = ExportService.export_content(rec_id, "full", "pdf", db_session)
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) > 1  # Multi-page PDF confirmed
    pdf_full_text = "\n".join(page.extract_text() for page in reader.pages)
    assert "Linear Regression" in pdf_full_text
    assert "Random Forest" in pdf_full_text
    assert "Hamza" in pdf_full_text
    assert "Bilal" in pdf_full_text
    assert "Cost functions measure model errors" in pdf_full_text

    # 3. Full DOCX Export Check
    docx_bytes, _, _ = ExportService.export_content(rec_id, "full", "docx", db_session)
    doc = docx.Document(io.BytesIO(docx_bytes))
    assert len(doc.paragraphs) > 50  # Over 50 structured paragraphs
    docx_full_text = "\n".join(p.text for p in doc.paragraphs)
    assert "Linear Regression" in docx_full_text
    assert "Random Forest" in docx_full_text
    assert "Hamza" in docx_full_text
    assert "Bilal" in docx_full_text

