import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import (
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_CONDITION_ON_PREVIOUS_TEXT,
    WHISPER_LOGPROB_THRESHOLD,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_TEMPERATURE_FALLBACK,
)
from app.models import (
    Base,
    Chunk,
    Recording,
    RecordingStatus,
    TranscriptASRAudit,
    TranscriptSegment,
)
from app.services.transcription_service import TranscriptionService


def _make_sqlite_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def _fake_refiner_keep_all(segments, **kwargs):
    return [
        SimpleNamespace(
            segment_index=s["segment_index"],
            refined_text=s["text"],
            was_pruned=False,
            changes_made=[],
        )
        for s in segments
    ]


def test_t5_whisper_decoding_parameters_and_preserved_thresholds(tmp_path):
    """T5 must enforce condition_on_previous_text=False, temperature=(0.0, 0.2), and preserve thresholds & language."""
    assert WHISPER_CONDITION_ON_PREVIOUS_TEXT is False
    assert WHISPER_TEMPERATURE_FALLBACK == (0.0, 0.2)
    assert WHISPER_COMPRESSION_RATIO_THRESHOLD == 2.2
    assert WHISPER_LOGPROB_THRESHOLD == -1.0
    assert WHISPER_NO_SPEECH_THRESHOLD == 0.6

    db = _make_sqlite_session()
    try:
        chunk_file = tmp_path / "chunk_0.wav"
        chunk_file.write_bytes(b"RIFF")

        rec = Recording(
            id="rec-t5-params",
            topic="Process Scheduling",
            filename="os.mp3",
            file_path=str(tmp_path / "os.mp3"),
            file_format=".mp3",
            media_type="audio",
            file_size_bytes=1024,
            duration_seconds=25.0,
            status=RecordingStatus.CHUNKED,
        )
        chk = Chunk(
            id="chk-t5-0",
            recording_id="rec-t5-params",
            chunk_index=0,
            start_time=0.0,
            end_time=25.0,
            duration=25.0,
            file_path=str(chunk_file),
        )
        db.add(rec)
        db.add(chk)
        db.commit()

        captured_kwargs = {}

        class FakeWhisperModel:
            dims = SimpleNamespace(n_text_ctx=448)
            is_multilingual = True

            def transcribe(self, **kwargs):
                captured_kwargs.update(kwargs)
                return {
                    "language": "en",
                    "text": "Process scheduling allocates CPU time.",
                    "segments": [
                        {
                            "id": 0,
                            "start": 1.0,
                            "end": 5.0,
                            "text": "Process scheduling allocates CPU time.",
                            "avg_logprob": -0.25,
                            "no_speech_prob": 0.02,
                            "compression_ratio": 1.1,
                            "temperature": 0.0,
                        }
                    ],
                }

        fake_audio = np.zeros(16000, dtype=np.float32)
        with (
            patch("app.services.transcription_service.WhisperModelManager.get_model", return_value=FakeWhisperModel()),
            patch("app.services.transcription_service.sf.read", return_value=(fake_audio, 16000)),
            patch("app.services.transcription_service.refine_transcript_segments", side_effect=_fake_refiner_keep_all),
            patch(
                "app.services.transcription_service.TranscriptionService._nlp_pipeline",
                side_effect=lambda txt, **kw: SimpleNamespace(normalized=txt),
            ),
            patch(
                "app.services.text_processor.TextProcessingService.translate_segments_batch",
                side_effect=lambda segs: None,
            ),
        ):
            TranscriptionService.transcribe_recording("rec-t5-params", db, model_size="base")

        assert captured_kwargs["condition_on_previous_text"] is False
        assert captured_kwargs["temperature"] == (0.0, 0.2)
        assert captured_kwargs["compression_ratio_threshold"] == 2.2
        assert captured_kwargs["logprob_threshold"] == -1.0
        assert captured_kwargs["no_speech_threshold"] == 0.6
        assert "language" not in captured_kwargs
    finally:
        db.close()


def test_t5_preserves_raw_whisper_and_records_all_rejected_segments(tmp_path):
    """T5 preserves untouched raw Whisper chunk output and logs all dropped/pruned segments with provenance."""
    db = _make_sqlite_session()
    try:
        chunk_paths = []
        for idx in range(3):
            p = tmp_path / f"chunk_{idx}.wav"
            p.write_bytes(b"RIFF")
            chunk_paths.append(p)

        rec = Recording(
            id="rec-t5-audit",
            topic="CPU Scheduling",
            filename="os.mp3",
            file_path=str(tmp_path / "os.mp3"),
            file_format=".mp3",
            media_type="audio",
            file_size_bytes=1024,
            duration_seconds=75.0,
            status=RecordingStatus.CHUNKED,
        )
        db.add(rec)
        for idx, p in enumerate(chunk_paths):
            db.add(
                Chunk(
                    id=f"chk-id-{idx}",
                    recording_id="rec-t5-audit",
                    chunk_index=idx,
                    start_time=float(idx * 25.0),
                    end_time=float((idx + 1) * 25.0),
                    duration=25.0,
                    file_path=str(p),
                )
            )
        db.commit()

        # Chunk 0: silent / no_speech_threshold skip (no segments returned)
        # Chunk 1:
        #   - hallucinated noise (",") -> rejected: hallucinated_noise
        #   - single-char junk ("a") -> rejected: empty_or_too_short_after_scrub
        #   - speech pruned by contextual refiner ("System noise blip") -> rejected: contextual_refiner_pruned
        #   - real speech with domain-correctable typo ("cpu sceduling uses processes") -> kept, raw_text preserved
        # Chunk 2:
        #   - pure boundary overlap token ("processes") -> pruned to empty by ContextStitcher -> rejected: stitch_overlap_pruned
        raw_chunk_outputs = [
            {
                "language": "en",
                "text": "",
                "segments": [],
            },
            {
                "language": "en",
                "text": ", a System noise blip cpu sceduling uses processes",
                "segments": [
                    {
                        "id": 0,
                        "start": 0.5,
                        "end": 3.0,
                        "text": ",",
                        "avg_logprob": -1.45,
                        "no_speech_prob": 0.72,
                        "compression_ratio": 1.2,
                        "temperature": 0.2,
                    },
                    {
                        "id": 1,
                        "start": 3.2,
                        "end": 4.0,
                        "text": "a",
                        "avg_logprob": -1.1,
                        "no_speech_prob": 0.4,
                        "compression_ratio": 0.8,
                        "temperature": 0.0,
                    },
                    {
                        "id": 2,
                        "start": 4.5,
                        "end": 10.0,
                        "text": "System noise blip",
                        "avg_logprob": -1.8,
                        "no_speech_prob": 0.55,
                        "compression_ratio": 1.9,
                        "temperature": 0.2,
                    },
                    {
                        "id": 3,
                        "start": 10.5,
                        "end": 18.0,
                        "text": "cpu sceduling uses processes",
                        "avg_logprob": -0.35,
                        "no_speech_prob": 0.05,
                        "compression_ratio": 1.15,
                        "temperature": 0.0,
                    },
                ],
            },
            {
                "language": "en",
                "text": "processes",
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 1.2,
                        "text": "processes",
                        "avg_logprob": -0.5,
                        "no_speech_prob": 0.1,
                        "compression_ratio": 1.0,
                        "temperature": 0.0,
                    }
                ],
            },
        ]
        call_counter = {"idx": 0}

        class FakeWhisperModel:
            dims = SimpleNamespace(n_text_ctx=448)
            is_multilingual = True

            def transcribe(self, **kwargs):
                res = raw_chunk_outputs[call_counter["idx"]]
                call_counter["idx"] += 1
                return res

        def fake_refiner(segments, **kwargs):
            results = []
            for s in segments:
                if "System noise blip" in s["text"]:
                    results.append(
                        SimpleNamespace(
                            segment_index=s["segment_index"],
                            refined_text="",
                            was_pruned=True,
                            changes_made=["pruned: noise blip"],
                        )
                    )
                else:
                    results.append(
                        SimpleNamespace(
                            segment_index=s["segment_index"],
                            refined_text=s["text"],
                            was_pruned=False,
                            changes_made=[],
                        )
                    )
            return results

        fake_audio = np.zeros(16000, dtype=np.float32)
        with (
            patch("app.services.transcription_service.WhisperModelManager.get_model", return_value=FakeWhisperModel()),
            patch("app.services.transcription_service.sf.read", return_value=(fake_audio, 16000)),
            patch("app.services.transcription_service.refine_transcript_segments", side_effect=fake_refiner),
            patch(
                "app.services.transcription_service.TranscriptionService._nlp_pipeline",
                side_effect=lambda txt, **kw: SimpleNamespace(normalized=txt),
            ),
            patch(
                "app.services.text_processor.TextProcessingService.translate_segments_batch",
                side_effect=lambda segs: None,
            ),
        ):
            result = TranscriptionService.transcribe_recording("rec-t5-audit", db, model_size="base")

        # 1) Untouched raw Whisper output preserved in return value and DB audit row
        assert len(result.raw_whisper_chunks) == 3
        assert result.raw_whisper_chunks[0]["chunk_id"] == "chk-id-0"
        assert result.raw_whisper_chunks[0]["segment_count"] == 0
        raw_chunk1_segs = result.raw_whisper_chunks[1]["segments"]
        assert [s["raw_text"] for s in raw_chunk1_segs] == [
            ",",
            "a",
            "System noise blip",
            "cpu sceduling uses processes",
        ]

        # Kept segment preserves verbatim raw_text even when domain corrector adjusted text
        kept_segs = db.query(TranscriptSegment).filter(TranscriptSegment.recording_id == "rec-t5-audit").all()
        assert len(kept_segs) == 1
        assert result.segments[1].raw_text == "cpu sceduling uses processes"

        # 2) All rejection mechanisms recorded with timestamps, raw_text, chunk_id, rejection_reason
        reasons = {r["rejection_reason"]: r for r in result.rejected_segments}
        assert "whisper_no_speech_or_empty" in reasons
        assert reasons["whisper_no_speech_or_empty"]["chunk_id"] == "chk-id-0"
        assert reasons["whisper_no_speech_or_empty"]["start_time"] == 0.0
        assert reasons["whisper_no_speech_or_empty"]["end_time"] == 25.0

        assert "hallucinated_noise" in reasons
        assert reasons["hallucinated_noise"]["chunk_id"] == "chk-id-1"
        assert reasons["hallucinated_noise"]["raw_text"] == ","
        assert reasons["hallucinated_noise"]["start_time"] == 25.5
        assert reasons["hallucinated_noise"]["end_time"] == 28.0

        assert "empty_or_too_short_after_scrub" in reasons
        assert reasons["empty_or_too_short_after_scrub"]["chunk_id"] == "chk-id-1"
        assert reasons["empty_or_too_short_after_scrub"]["raw_text"] == "a"
        assert reasons["empty_or_too_short_after_scrub"]["start_time"] == 28.2

        assert "stitch_overlap_pruned" in reasons
        assert reasons["stitch_overlap_pruned"]["chunk_id"] == "chk-id-2"
        assert reasons["stitch_overlap_pruned"]["raw_text"] == "processes"
        assert reasons["stitch_overlap_pruned"]["start_time"] == 50.0

        pruned_key = next(k for k in reasons if k.startswith("contextual_refiner_pruned"))
        assert reasons[pruned_key]["chunk_id"] == "chk-id-1"
        assert reasons[pruned_key]["raw_text"] == "System noise blip"
        assert reasons[pruned_key]["start_time"] == 29.5
        assert reasons[pruned_key]["end_time"] == 35.0

        # 3) DB audit persistence verified & cleanup works on re-transcribe
        audit_row = (
            db.query(TranscriptASRAudit)
            .filter(TranscriptASRAudit.recording_id == "rec-t5-audit")
            .first()
        )
        assert audit_row is not None
        persisted_raw = json.loads(audit_row.raw_whisper_chunks_json)
        persisted_rejected = json.loads(audit_row.rejected_segments_json)
        assert persisted_raw[1]["segments"][3]["raw_text"] == "cpu sceduling uses processes"
        assert len(persisted_rejected) == 5

        TranscriptionService.cleanup_transcripts("rec-t5-audit", db)
        assert db.query(TranscriptASRAudit).filter(TranscriptASRAudit.recording_id == "rec-t5-audit").count() == 0
    finally:
        db.close()
