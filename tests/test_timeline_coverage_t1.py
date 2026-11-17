import json
from pathlib import Path

import pytest

from app.services.topic_intelligence import (
    LectureStudyNotes,
    Segment,
    StudyNotesQualityValidator,
    StudyTopic,
    TopicSpan,
    compute_timeline_coverage,
)


def test_overlapping_intervals_union():
    """Overlapping intervals must be merged, never double-counted."""
    cov = compute_timeline_coverage([(0.0, 20.0), (10.0, 30.0)], actual_duration=50.0)
    assert cov["covered_seconds"] == pytest.approx(30.0, abs=0.01)
    assert cov["uncovered_seconds"] == pytest.approx(20.0, abs=0.01)
    assert cov["coverage_pct"] == pytest.approx(60.0, abs=0.01)
    assert cov["interior_gaps"] == []
    assert cov["tail_gap"] == {"start": 30.0, "end": 50.0, "duration": 20.0}


def test_adjacent_intervals_no_false_gap():
    """Adjacent intervals [0, 10] and [10, 20] must have 0 interior gaps."""
    cov = compute_timeline_coverage([(0.0, 10.0), (10.0, 20.0)], actual_duration=20.0)
    assert cov["covered_seconds"] == pytest.approx(20.0, abs=0.01)
    assert cov["coverage_pct"] == pytest.approx(100.0, abs=0.01)
    assert cov["interior_gaps"] == []
    assert cov["tail_gap"] is None


def test_internal_gaps_threshold_5s():
    """Only interior gaps strictly longer than 5 seconds are reported."""
    cov = compute_timeline_coverage(
        [(0.0, 20.0), (24.5, 40.0), (52.0, 80.0)],
        actual_duration=100.0,
    )
    assert cov["covered_seconds"] == pytest.approx(20.0 + 15.5 + 28.0, abs=0.01)
    assert len(cov["interior_gaps"]) == 1
    assert cov["interior_gaps"][0] == {"start": 40.0, "end": 52.0, "duration": 12.0}
    assert cov["tail_gap"] == {"start": 80.0, "end": 100.0, "duration": 20.0}


def test_validator_uses_actual_duration_and_union_not_last_timestamp():
    """Validator must not report 100% coverage when interior/tail gaps exist."""
    segs = [
        Segment(id=1, start=0.0, end=140.0, text="Process management and CPU scheduling basics."),
        Segment(id=2, start=175.0, end=235.0, text="Ready queue transitions and turnaround time."),
        Segment(id=3, start=260.0, end=472.7, text="Shortest job first and aging to prevent starvation."),
    ]
    topics = [
        StudyTopic(
            id="t1",
            title="CPU Scheduling Basics",
            spans=[TopicSpan(start=0.0, end=235.0, start_seg_id=1, end_seg_id=2)],
            explanation="CPU scheduling selects which ready process runs next on the processor.",
            key_points=["Process management and CPU scheduling basics."],
            source_segment_ids=[1, 2],
            source_text="Process management and CPU scheduling basics. Ready queue transitions.",
        ),
        StudyTopic(
            id="t2",
            title="SJF and Aging",
            spans=[TopicSpan(start=260.0, end=472.7, start_seg_id=3, end_seg_id=3)],
            explanation="Shortest job first reduces waiting time while aging prevents starvation.",
            key_points=["Shortest job first and aging to prevent starvation."],
            source_segment_ids=[3],
            source_text="Shortest job first and aging to prevent starvation.",
        ),
    ]
    notes = LectureStudyNotes(
        study_topics=topics,
        lecture_overview="OS lecture overview.",
        subject="Operating System",
        theme="CPU Scheduling",
    )
    validator = StudyNotesQualityValidator()
    _passed, _score, details = validator.validate(
        notes, duration=510.95, segments=segs, speech_duration_seconds=316.81
    )
    tc = details["timeline_coverage"]
    assert tc["duration"] == pytest.approx(510.95, abs=0.1)
    assert tc["latest_end"] == pytest.approx(472.7, abs=0.1)
    assert tc["coverage_pct"] < 85.0
    assert len(tc["interior_gaps"]) == 2
    assert tc["tail_gap"] is not None
    assert tc["tail_gap_seconds"] == pytest.approx(38.25, abs=0.1)
    assert "audible_speech_coverage_pct" not in tc


def test_invalid_audible_speech_coverage_pct_not_reported():
    """Scalar-divided audible_speech_coverage_pct must not be reported without VAD interval intersection."""
    cov = compute_timeline_coverage(
        [(0.0, 410.22)],
        actual_duration=510.91,
        speech_duration_seconds=316.81,
    )
    assert "audible_speech_coverage_pct" not in cov


def test_reference_recording_baseline_coverage_and_gaps():
    """Verify T1 coverage metrics against the captured T0 reference baseline."""
    base = Path("audit_baseline")
    if not (base / "persisted_transcript.json").exists():
        pytest.skip("audit_baseline artifacts not present")

    rec = json.loads((base / "recording_and_media.json").read_text(encoding="utf-8"))["recording"]
    segs = json.loads((base / "persisted_transcript.json").read_text(encoding="utf-8"))["segments"]
    intervals = [(s["start_time"], s["end_time"]) for s in segs]

    cov_persisted = compute_timeline_coverage(
        intervals,
        actual_duration=float(rec["duration_seconds"]),
        speech_duration_seconds=float(rec["speech_duration_seconds"]),
        gap_threshold=5.0,
    )
    assert cov_persisted["actual_duration"] == pytest.approx(510.91, abs=0.05)
    assert cov_persisted["last_transcript_timestamp"] == pytest.approx(476.72, abs=0.05)
    assert cov_persisted["coverage_pct"] == pytest.approx(80.29, abs=0.1)
    assert cov_persisted["tail_gap"] is not None
    assert cov_persisted["tail_gap_seconds"] == pytest.approx(34.19, abs=0.1)
    assert len(cov_persisted["interior_gaps"]) == 3
    assert "audible_speech_coverage_pct" not in cov_persisted

    iv = json.loads((base / "intelligence_view.json").read_text(encoding="utf-8"))
    trusted_iv = [s for s in iv["segments_json"] if s.get("trust") == "high"]
    cov_high_trust = compute_timeline_coverage(
        [(s["start_time"], s["end_time"]) for s in trusted_iv],
        actual_duration=float(rec["duration_seconds"]),
        speech_duration_seconds=float(rec["speech_duration_seconds"]),
        gap_threshold=5.0,
    )
    assert cov_high_trust["last_transcript_timestamp"] == pytest.approx(472.72, abs=0.05)
    assert cov_high_trust["coverage_pct"] == pytest.approx(49.38, abs=0.1)
    assert cov_high_trust["tail_gap_seconds"] == pytest.approx(38.19, abs=0.1)
    assert len(cov_high_trust["interior_gaps"]) == 5
    assert "audible_speech_coverage_pct" not in cov_high_trust
