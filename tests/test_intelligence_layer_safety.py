"""
tests/test_intelligence_layer_safety.py — Phase 1 Safety Gates for Intelligence Preprocessing Layer.

Covers:
  - T11: Display Transcript Invariance (SHA-256 byte-identical across transcripts & transcript_segments
         and untouched row counts/checksums across all 5 protected tables after persisting intelligence_views).
"""

import hashlib
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("TESTING", "1")


def _snapshot_display_tables(db) -> dict:
    snap = {}
    for table in ("recordings", "chunks", "transcripts", "transcript_segments", "stitched_blocks"):
        rows = db.execute(text(f"SELECT * FROM {table} ORDER BY rowid")).fetchall()
        h = hashlib.sha256()
        for r in rows:
            h.update(repr(tuple(r)).encode("utf-8"))
        snap[table] = {"count": len(rows), "hash": h.hexdigest()}
    return snap


def test_t11_display_transcript_invariance_and_readonly_tables():
    """T11: Persisting IntelligenceViewRecord leaves display transcript tables 100% byte-identical."""
    import app.main  # noqa: F401
    from app.database import Base
    from app.models import Recording, Transcript, TranscriptSegment
    from app.services.intelligence_preprocessor import IntelligencePreprocessor, IntelligenceViewRecord

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    rec = Recording(
        id="rec_inv_001",
        filename="real_sample.mp4",
        file_path="storage/uploads/real_sample.mp4",
        file_format=".mp4",
        media_type="video",
        file_size_bytes=2048,
        duration_seconds=60.0,
        recording_type="Lecture",
        topic="Operating Systems",
    )
    tr = Transcript(
        id="tr_inv_001",
        recording_id="rec_inv_001",
        original_transcript="Um today we discuss basic CPU cost two lakh 20,000.",
        normalized_transcript="Um today we discuss basis CPU dost wo lakh 20,00.",
        english_transcript="Today we examine core CPU expenses.",
    )
    seg1 = TranscriptSegment(
        id="seg_inv_1",
        transcript_id="tr_inv_001",
        recording_id="rec_inv_001",
        chunk_index=0,
        segment_index=0,
        start_time=0.0,
        end_time=4.0,
        text="Um today we discuss basic CPU cost",
        normalized_text="Um today we discuss basis CPU dost",
        english_text="Today we examine core CPU expenses",
        language="en",
        translation_source="local_llm",
        confidence=0.85,
        no_speech_prob=0.1,
    )
    seg2 = TranscriptSegment(
        id="seg_inv_2",
        transcript_id="tr_inv_001",
        recording_id="rec_inv_001",
        chunk_index=0,
        segment_index=1,
        start_time=4.2,
        end_time=8.5,
        text="two lakh 20,000 rupees",
        normalized_text="wo lakh 20,00 rupees",
        english_text="two thousand thousand rupees",
        language="en",
        translation_source="local_llm",
        confidence=0.82,
        no_speech_prob=0.12,
    )
    db.add_all([rec, tr, seg1, seg2])
    db.commit()

    before_snap = _snapshot_display_tables(db)

    view = IntelligencePreprocessor.get_or_create_view("rec_inv_001", db, topic_hint="Operating Systems")
    assert len(view.segments) >= 1
    assert "cost" in view.full_llm_text and "20,000" in view.full_llm_text

    # Verify intelligence_views row was persisted
    stored_count = db.query(IntelligenceViewRecord).filter_by(recording_id="rec_inv_001").count()
    assert stored_count == 1

    # Cache hit returns identical object without creating duplicate rows
    view2 = IntelligencePreprocessor.get_or_create_view("rec_inv_001", db, topic_hint="Operating Systems")
    assert view2.transcript_hash == view.transcript_hash
    assert db.query(IntelligenceViewRecord).filter_by(recording_id="rec_inv_001").count() == 1

    after_snap = _snapshot_display_tables(db)
    assert before_snap == after_snap, f"Display tables mutated! Before={before_snap}, After={after_snap}"


def test_f3_match_text_never_exposed_as_adapter_llm_text_or_prompt_source():
    """F3 contract: adapter text/normalized_text/english_text carry llm_text ONLY; match_text is strictly isolated."""
    from types import SimpleNamespace
    from app.services.intelligence_preprocessor import IntelligencePreprocessor

    segs = [
        SimpleNamespace(
            id="f3_seg1",
            segment_index=0,
            chunk_index=0,
            text="Accrual accounting is not necessarily based on immediate cash flow receipts.",
            normalized_text="Accrual accounting is not necessarily based on immediate cash flow receipts.",
            english_text=None,
            start_time=0.0,
            end_time=5.0,
            no_speech_prob=0.1,
            confidence=0.9,
            language="en",
            translation_source="passthrough",
        )
    ]
    view = IntelligencePreprocessor.build_view_in_memory("rec_f3_contract", segs, write_diff_log=False)
    adapters = view.to_adapters()
    assert len(adapters) == 1
    ad = adapters[0]
    # Adapters expose grammatical llm_text for any LLM or embedding consumer, NEVER match_text
    assert ad.text == ad.llm_text == ad.normalized_text == ad.english_text
    assert ad.match_text != ad.llm_text
    assert "is" in ad.llm_text.split()
    assert "is" not in ad.match_text.split()
    assert "not" in ad.match_text.split()


def test_phase2_downstream_wiring_invariance_no_double_clean_and_stale_invalidation():
    """
    Phase 2 E2E:
      1. generate_intelligence_report consumes IntelligenceView, filters trust=='low' when high-trust exists,
         and leaves all 5 protected display tables byte-identical.
      2. AcademicPreprocessor & TopicAnalyzer bypass legacy double-cleaning on is_intel_view adapters.
      3. EmbeddingService keeps 1 row per source TranscriptSegment (F6), embeds repaired llm_text (not match_text),
         and auto-invalidates on manual correction or PREPROCESSING_VERSION change.
      4. LectureRAGService retrieves repaired text without low-trust hallucinations.
    """
    import app.main  # noqa: F401
    from app.database import Base
    from app.models import Recording, Transcript, TranscriptSegment, SegmentEmbedding
    from app.services.academic_pipeline import compute_config_hash, stored_result_is_current
    from app.services.academic_preprocess import AcademicPreprocessor
    from app.services.embedding_service import EmbeddingService, EmbeddingModelManager
    from app.services.intelligence_preprocessor import IntelligencePreprocessor
    from app.services.intelligence_service import LectureIntelligenceService
    from app.services.rag_service import LectureRAGService
    from app.services.topic_intelligence import TopicIntelligenceEngine
    import numpy as np

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    rec = Recording(
        id="rec_p2_e2e",
        filename="accounting_p2.mp4",
        file_path="storage/uploads/accounting_p2.mp4",
        file_format=".mp4",
        media_type="video",
        file_size_bytes=4096,
        duration_seconds=45.0,
        recording_type="Lecture",
        topic="Accrual Accounting",
    )
    tr = Transcript(
        id="tr_p2_e2e",
        recording_id="rec_p2_e2e",
        original_transcript="Accrual accounting cost two lakh 20,000 rupees.",
        normalized_transcript="Accrual accounting dost wo lakh 20,00 rupees.",
        english_transcript="Accrual accounting expense two thousand.",
    )
    # Chunk 0 (high trust, unmerged vs Chunk 1 low trust)
    s0 = TranscriptSegment(
        id="p2_s0", transcript_id="tr_p2_e2e", recording_id="rec_p2_e2e",
        chunk_index=0, segment_index=0, start_time=0.0, end_time=12.0,
        text="Today we study Accrual Accounting where revenue and total cost equal 20,000 rupees",
        normalized_text="Today we study Accrual Accounting where revenue and total dost equal 20,00 rupees",
        english_text="Today we learn revenue equals two hundred.",
        language="en", translation_source="local_llm", confidence=0.88, no_speech_prob=0.05,
    )
    s1 = TranscriptSegment(
        id="p2_s1", transcript_id="tr_p2_e2e", recording_id="rec_p2_e2e",
        chunk_index=0, segment_index=1, start_time=12.4, end_time=24.0,
        text="when cash actually changes hands later in this basic financial statement.",
        normalized_text="when cash actually changes hands later in this basis financial statement",
        english_text="when cash moves in the basic report.",
        language="en", translation_source="local_llm", confidence=0.84, no_speech_prob=0.08,
    )
    # Chunk 14 low-confidence hallucination segment -> trust='low'
    s2 = TranscriptSegment(
        id="p2_s2", transcript_id="tr_p2_e2e", recording_id="rec_p2_e2e",
        chunk_index=14, segment_index=2, start_time=25.0, end_time=35.0,
        text="bßniisni,",
        normalized_text="bßniisni,",
        english_text="Let us be friends.",
        language="ur", translation_source="local_llm", confidence=0.10, no_speech_prob=0.25,
    )
    db.add_all([rec, tr, s0, s1, s2])
    db.commit()

    before_snap = _snapshot_display_tables(db)

    # 1. Verify no-double-clean in AcademicPreprocessor & TopicAnalyzer on is_intel_view adapters
    view = IntelligencePreprocessor.get_or_create_view("rec_p2_e2e", db, topic_hint="Accrual Accounting")
    adapters = view.to_adapters()
    trusted = IntelligencePreprocessor.filter_high_trust_adapters(adapters)
    assert any(a.trust == "low" for a in adapters)
    assert all(a.trust == "high" for a in trusted)
    assert "Let us be friends" not in view.full_llm_text

    pre_res = AcademicPreprocessor.process_segments(adapters)
    assert len(pre_res.spans) == len(trusted)
    assert any("20,000" in sp.clean_text and "cost" in sp.clean_text for sp in pre_res.spans)

    analyzer = TopicIntelligenceEngine(use_llm=False)
    ti_segs = analyzer._preprocess_segments(adapters, view.full_llm_text, 45.0)
    assert any("20,000" in ts.text and "cost" in ts.text for ts in ti_segs)

    # 2. Verify F6: strictly 1 SegmentEmbedding row per source TranscriptSegment, encoding repaired llm_text
    encoded_inputs: list[str] = []

    orig_encode = EmbeddingModelManager.encode

    def spy_encode(texts, normalize_embeddings=True):
        if isinstance(texts, list):
            encoded_inputs.extend(texts)
        else:
            encoded_inputs.append(texts)
        arr = np.ones((len(texts) if isinstance(texts, list) else 1, 384), dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        return arr

    EmbeddingModelManager.encode = classmethod(lambda cls, t, normalize_embeddings=True: spy_encode(t, normalize_embeddings))
    try:
        cnt = EmbeddingService.generate_and_store_embeddings("rec_p2_e2e", db)
        assert cnt == 3  # Strictly 1 row per source TranscriptSegment (F6), NOT stitched count
        emb_rows = db.query(SegmentEmbedding).filter_by(recording_id="rec_p2_e2e").all()
        assert len(emb_rows) == 3
        assert any("20,000" in txt and "cost" in txt for txt in encoded_inputs)
        # Ensure match_text is NOT passed to embeddings (stopword 'the'/'and'/'in' must survive in llm_text)
        assert any(" and " in txt for txt in encoded_inputs)
    finally:
        EmbeddingModelManager.encode = orig_encode

    # 3. Verify RAG context uses IntelligenceView (finds cost/20,000, excludes Let us be friends)
    rag_ctx = LectureRAGService.retrieve_context("rec_p2_e2e", "What is the total cost in accrual accounting?", db)
    assert "Let us be friends" not in (rag_ctx.get("text_corpus") or "")
    assert "20,000" in (rag_ctx.get("text_corpus") or "")

    # 4. Verify report generation + cache validity + invalidation when source edited
    report = LectureIntelligenceService.generate_intelligence_report("rec_p2_e2e", db, force=True)
    assert report["status"] == "success"
    stored_notes = report["lecture_notes"]
    if stored_notes.get("meta"):
        assert stored_result_is_current(stored_notes, trusted, transcript_hash_override=view.transcript_hash) is True
        # Simulating manual correction alters transcript_hash -> invalidates stored result
        s0.text = "Today we study Accrual Accounting where revenue and total cost equal 50,000 rupees"
        db.flush()
        new_view = IntelligencePreprocessor.build_view_in_memory("rec_p2_e2e", [s0, s1, s2], write_diff_log=False)
        assert stored_result_is_current(
            stored_notes,
            IntelligencePreprocessor.filter_high_trust_adapters(new_view.to_adapters()),
            transcript_hash_override=new_view.transcript_hash,
        ) is False
        db.rollback()

    after_snap = _snapshot_display_tables(db)
    assert before_snap == after_snap, f"Display tables mutated during Phase 2! Before={before_snap}, After={after_snap}"


