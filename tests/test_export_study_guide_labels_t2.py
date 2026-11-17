import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.export_service import ExportService


def _make_export_data(quality_pass: bool, is_fallback: bool, high_trust_cov: float = 49.38):
    rec = SimpleNamespace(
        filename="operating system lecture.mp4",
        duration_seconds=510.911,
        language_summary="hi: 86%, en: 14%",
        recording_type="Lecture",
    )
    segments = [
        SimpleNamespace(
            start_time=0.0,
            end_time=410.22,
            text="CPU scheduling lecture segment.",
            normalized_text="CPU scheduling lecture segment.",
            english_text="CPU scheduling lecture segment.",
            language="en",
        )
    ]
    notes = {
        "analyzer_used": "local_extractive_fallback" if is_fallback else "local_llm",
        "is_fallback": is_fallback,
        "quality_pass": quality_pass,
        "quality_details": {
            "timeline_coverage": {
                "coverage_pct": high_trust_cov,
                "ratio": round(high_trust_cov / 100.0, 4),
                "pass": high_trust_cov >= 80.0,
            }
        },
        "lecture_overview": "Overview of CPU scheduling.",
        "study_topics": [
            {
                "title": "Process Scheduling",
                "spans": [{"start": 0.0, "end": 100.0}],
                "explanation": "CPU scheduler selects a ready process.",
                "key_points": ["CPU scheduler selects a ready process."],
                "definitions": [],
                "examples": [],
                "formulas": [],
                "emphasis": [],
                "exam_relevant": False,
            }
        ],
        "markdown_study_guide": "### 1. The Main Idea\nCPU scheduling selects ready processes.",
    }
    return {
        "recording": rec,
        "transcript": SimpleNamespace(original_transcript="...", normalized_transcript="...", english_transcript="..."),
        "segments": segments,
        "summary_short": "Short summary.",
        "summary_detailed": "Detailed summary.",
        "lecture_notes": notes,
        "assignments": [],
        "action_items": [],
        "quizzes": [],
    }


@pytest.mark.parametrize(
    "quality_pass,is_fallback,expect_high_yield",
    [
        (True, False, True),
        (False, False, False),
        (True, True, False),
        (False, True, False),
    ],
)
def test_markdown_high_yield_label_all_combinations(quality_pass, is_fallback, expect_high_yield):
    """High-Yield label is permitted ONLY when quality_pass=True and is_fallback=False."""
    high_trust_cov = 92.0 if expect_high_yield else 49.38
    data = _make_export_data(quality_pass=quality_pass, is_fallback=is_fallback, high_trust_cov=high_trust_cov)
    md = ExportService._build_markdown(data, content_type="notes")

    if expect_high_yield:
        assert "## High-Yield Master Study Guide (14 Sections)" in md
        assert "Generated from limited or low-confidence transcript; review before use." not in md
    else:
        assert "High-Yield Master Study Guide" not in md
        assert "## Lecture Study Guide" in md
        assert "Generated from limited or low-confidence transcript; review before use." in md
        assert "Persisted transcript temporal coverage:** 80.29%" in md
        assert "High-trust evidence coverage used by notes validation:** 49.38% (below the 80% quality threshold)" in md


def test_pdf_export_suppresses_high_yield_on_fallback():
    """PDF export must not use High-Yield Master Study Guide & Notes when fallback or failed."""
    data_fail = _make_export_data(quality_pass=False, is_fallback=True, high_trust_cov=49.38)
    pdf_fail = ExportService._build_pdf(data_fail, content_type="notes")
    assert isinstance(pdf_fail, bytes) and len(pdf_fail) > 500
    assert ExportService._evaluate_study_guide_quality(data_fail)["is_high_yield"] is False

    data_ok = _make_export_data(quality_pass=True, is_fallback=False, high_trust_cov=90.0)
    pdf_ok = ExportService._build_pdf(data_ok, content_type="notes")
    assert isinstance(pdf_ok, bytes) and len(pdf_ok) > 500
    assert ExportService._evaluate_study_guide_quality(data_ok)["is_high_yield"] is True


def test_reference_baseline_export_suppresses_high_yield_and_shows_both_coverages():
    """Verify T2 against the captured T0 reference baseline with T1 high-trust coverage."""
    base = Path("audit_baseline")
    if not (base / "educational_outputs.json").exists():
        pytest.skip("audit_baseline artifacts not present")

    rec_d = json.loads((base / "recording_and_media.json").read_text(encoding="utf-8"))["recording"]
    segs_d = json.loads((base / "persisted_transcript.json").read_text(encoding="utf-8"))["segments"]
    eo = json.loads((base / "educational_outputs.json").read_text(encoding="utf-8"))

    notes = dict(eo["lecture_notes"])
    # Simulate T1 validated timeline_coverage on the 37 high-trust segments (49.38%)
    notes["quality_details"] = dict(notes.get("quality_details") or {})
    notes["quality_details"]["timeline_coverage"] = {
        "coverage_pct": 49.38,
        "ratio": 0.4938,
        "pass": False,
    }

    data = {
        "recording": SimpleNamespace(**rec_d),
        "transcript": SimpleNamespace(original_transcript="", normalized_transcript="", english_transcript=""),
        "segments": [SimpleNamespace(**s) for s in segs_d],
        "summary_short": eo["summary_short"],
        "summary_detailed": eo["summary_detailed"],
        "lecture_notes": notes,
        "assignments": eo["assignments"],
        "action_items": eo["action_items"],
        "quizzes": eo["quizzes"],
    }
    md = ExportService._build_markdown(data, content_type="full")
    assert "High-Yield Master Study Guide" not in md
    assert "## Lecture Study Guide" in md
    assert "Generated from limited or low-confidence transcript; review before use." in md
    assert "Persisted transcript temporal coverage:** 80.29%" in md
    assert "High-trust evidence coverage used by notes validation:** 49.38% (below the 80% quality threshold)" in md
