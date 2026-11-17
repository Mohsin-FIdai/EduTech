"""
export_service.py — Multi-format export service (PDF, DOCX, Plain Text, Markdown)
for Lecture & Meeting Intelligence reports and transcripts.
"""

import io
import re
import json
import html
import logging
from typing import Optional
from sqlalchemy.orm import Session

from app.models import Recording, Transcript, TranscriptSegment, IntelligenceReport
from app.services.intelligence_service import LectureIntelligenceService

logger = logging.getLogger(__name__)


class ExportException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ExportService:
    """
    Generates downloadable reports in Markdown, Plain Text, DOCX, and PDF.
    Supports full report and selective export (transcript, summary, notes, assignments, action_items, quizzes).
    """

    @classmethod
    def get_recording_and_data(
        cls, recording_id: str, db: Session
    ) -> dict:
        """
        Retrieves or generates intelligence data and transcript for a recording.
        """
        recording = db.query(Recording).filter(Recording.id == recording_id).first()
        if not recording:
            raise ExportException(f"Recording '{recording_id}' not found.")

        transcript = db.query(Transcript).filter(Transcript.recording_id == recording_id).first()
        segments = (
            db.query(TranscriptSegment)
            .filter(TranscriptSegment.recording_id == recording_id)
            .order_by(TranscriptSegment.start_time.asc())
            .all()
        )

        report = db.query(IntelligenceReport).filter(IntelligenceReport.recording_id == recording_id).first()
        if not report:
            # Auto-generate if not present
            try:
                raw_data = LectureIntelligenceService.generate_intelligence_report(recording_id, db)
                summary_short = raw_data["summary_short"]
                summary_detailed = raw_data["summary_detailed"]
                notes_data = raw_data["lecture_notes"]
                assignments_data = raw_data["assignments"]
                action_items_data = raw_data["action_items"]
                quizzes_data = raw_data["quizzes"]
            except Exception as e:
                logger.warning(f"Failed to generate intelligence on the fly: {e}")
                summary_short = "Summary not available."
                summary_detailed = "Detailed summary not available."
                notes_data = {"topics": [], "subtopics": [], "definitions": [], "key_concepts": [], "important_points": []}
                assignments_data = []
                action_items_data = []
                quizzes_data = []
        else:
            summary_short = report.summary_short
            summary_detailed = report.summary_detailed
            try:
                notes_data = json.loads(report.lecture_notes_json)
            except Exception:
                notes_data = {"topics": [], "subtopics": [], "definitions": [], "key_concepts": [], "important_points": []}
            try:
                assignments_data = json.loads(report.assignments_json)
            except Exception:
                assignments_data = []
            try:
                action_items_data = json.loads(report.action_items_json)
            except Exception:
                action_items_data = []
            try:
                quizzes_data = json.loads(report.quizzes_json)
            except Exception:
                quizzes_data = []

        return {
            "recording": recording,
            "transcript": transcript,
            "segments": segments,
            "summary_short": summary_short,
            "summary_detailed": summary_detailed,
            "lecture_notes": notes_data,
            "assignments": assignments_data,
            "action_items": action_items_data,
            "quizzes": quizzes_data,
        }

    @classmethod
    def export_content(
        cls,
        recording_id: str,
        content_type: str = "full",
        format: str = "markdown",
        db: Optional[Session] = None
    ) -> tuple[bytes, str, str]:
        """
        Main export orchestrator.
        Returns: (file_bytes, filename, media_type)
        """
        if db is None:
            raise ExportException("Database session required for export.")

        valid_content_types = ["full", "transcript", "summary", "notes", "assignments", "action_items", "quizzes"]
        clean_content_type = content_type.lower().strip()
        if clean_content_type not in valid_content_types:
            raise ExportException(f"Invalid content_type '{content_type}'. Must be one of: {valid_content_types}")

        clean_format = format.lower().strip()
        if clean_format in ["md", "markdown"]:
            clean_format = "markdown"
        elif clean_format in ["txt", "text"]:
            clean_format = "text"
        elif clean_format in ["docx", "doc"]:
            clean_format = "docx"
        elif clean_format in ["pdf"]:
            clean_format = "pdf"
        else:
            raise ExportException(f"Invalid format '{format}'. Must be one of: markdown, text, docx, pdf")

        data = cls.get_recording_and_data(recording_id, db)
        rec_name = data["recording"].filename.rsplit(".", 1)[0] if data["recording"] else recording_id

        if clean_format == "markdown":
            content_str = cls._build_markdown(data, clean_content_type)
            file_bytes = content_str.encode("utf-8")
            filename = f"{rec_name}_{clean_content_type}.md"
            media_type = "text/markdown; charset=utf-8"
        elif clean_format == "text":
            content_str = cls._build_plain_text(data, clean_content_type)
            file_bytes = content_str.encode("utf-8")
            filename = f"{rec_name}_{clean_content_type}.txt"
            media_type = "text/plain; charset=utf-8"
        elif clean_format == "docx":
            file_bytes = cls._build_docx(data, clean_content_type)
            filename = f"{rec_name}_{clean_content_type}.docx"
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif clean_format == "pdf":
            file_bytes = cls._build_pdf(data, clean_content_type)
            filename = f"{rec_name}_{clean_content_type}.pdf"
            media_type = "application/pdf"
        else:
            raise ExportException(f"Unsupported format: {clean_format}")

        return file_bytes, filename, media_type

    # =========================================================================
    @classmethod
    def _evaluate_study_guide_quality(cls, data: dict) -> dict:
        """
        Determine whether study-guide output qualifies for the High-Yield label
        and compute distinct persisted-transcript vs high-trust coverage metrics.
        """
        notes = data.get("lecture_notes") or {}
        if "is_fallback" in notes:
            is_fallback = bool(notes.get("is_fallback"))
        else:
            is_fallback = notes.get("analyzer_used") == "local_extractive_fallback"
        quality_pass = bool(notes.get("quality_pass", False))
        is_high_yield = quality_pass and (not is_fallback)

        rec = data.get("recording")
        duration = float(getattr(rec, "duration_seconds", 0.0) or 0.0) if rec else 0.0
        segments = data.get("segments") or []

        persisted_cov_pct: Optional[float] = None
        if duration > 0 and segments:
            from app.services.topic_intelligence import compute_timeline_coverage
            cov = compute_timeline_coverage(
                [(float(getattr(s, "start_time", 0.0) or 0.0), float(getattr(s, "end_time", 0.0) or 0.0)) for s in segments],
                actual_duration=duration,
            )
            persisted_cov_pct = cov.get("coverage_pct")

        q_details = notes.get("quality_details") or {}
        tc = q_details.get("timeline_coverage") or {}
        high_trust_cov_pct: Optional[float] = None
        if tc.get("coverage_pct") is not None:
            high_trust_cov_pct = float(tc["coverage_pct"])
        elif tc.get("ratio") is not None:
            high_trust_cov_pct = round(float(tc["ratio"]) * 100.0, 2)

        return {
            "is_high_yield": is_high_yield,
            "quality_pass": quality_pass,
            "is_fallback": is_fallback,
            "persisted_coverage_pct": persisted_cov_pct,
            "high_trust_coverage_pct": high_trust_cov_pct,
            "warning": "Generated from limited or low-confidence transcript; review before use.",
        }

    # =========================================================================
    # 1. MARKDOWN BUILDER
    # =========================================================================
    @classmethod
    def _escape_md(cls, text: any) -> str:
        """
        Escapes Markdown structural and inline formatting characters
        (\\, *, _, `, <, >, |, #) when inserting dynamic content, preventing
        HTML tag omission, unintended italics/bold, or table/heading corruption.
        """
        if text is None:
            return ""
        s = str(text)
        # Escape backslash first
        s = s.replace('\\', r'\\')
        # Escape formatting, html tag delimiters, pipes, and header hashes
        replacements = [
            ('*', r'\*'),
            ('_', r'\_'),
            ('`', r'\`'),
            ('<', r'\<'),
            ('>', r'\>'),
            ('|', r'\|'),
            ('#', r'\#'),
        ]
        for char, repl in replacements:
            s = s.replace(char, repl)
        return s

    @classmethod
    def _build_markdown(cls, data: dict, content_type: str) -> str:
        rec = data["recording"]
        lines = []
        lines.append(f"# Lecture & Meeting Intelligence Report")
        lines.append(f"")
        lines.append(f"**Recording:** `{rec.filename if rec else 'Unknown'}`  ")
        lines.append(f"**Duration:** {rec.duration_seconds if rec else 0.0:.2f} seconds  ")
        lines.append(f"**Export Scope:** `{content_type}`  ")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

        # 1. Summary
        if content_type in ["full", "summary"]:
            lines.append(f"## Executive Summary")
            lines.append(f"")
            lines.append(f"### Short Summary")
            lines.append(f"{cls._escape_md(data['summary_short'])}")
            lines.append(f"")
            lines.append(f"### Detailed Summary")
            lines.append(f"{cls._escape_md(data['summary_detailed'])}")
            lines.append(f"")
            lines.append(f"---")
            lines.append(f"")

        # 2. Lecture Notes
        if content_type in ["full", "notes"]:
            notes = data["lecture_notes"]
            lines.append(f"## Structured Lecture Notes")
            lines.append(f"")

            # v2: Render study_topics if available
            study_topics = notes.get("study_topics", [])
            if study_topics:
                overview = notes.get("lecture_overview", "")
                analyzer = notes.get("analyzer_used", "local")
                if overview:
                    lines.append(f"### Lecture Overview")
                    lines.append(f"{cls._escape_md(overview)}")
                    lines.append(f"")
                lines.append(f"### Study Topics _(Source: {analyzer})_")
                lines.append(f"")
                for i, topic in enumerate(study_topics, 1):
                    title = cls._escape_md(topic.get("title", "Topic"))
                    spans = topic.get("spans", [])
                    if spans:
                        start = spans[0].get("start", 0)
                        end = spans[-1].get("end", 0)
                        ts = f"`[{start:.0f}s – {end:.0f}s]`"
                    else:
                        ts = ""
                    exam_tag = " **[EXAM RELEVANT]**" if topic.get("exam_relevant") else ""
                    lines.append(f"#### {i}. {title}{exam_tag} {ts}")
                    expl = topic.get("explanation", "")
                    if expl:
                        lines.append(f"")
                        lines.append(f"{cls._escape_md(expl)}")
                    key_points = topic.get("key_points", [])
                    if key_points:
                        lines.append(f"")
                        for kp in key_points[:6]:
                            lines.append(f"- {cls._escape_md(kp)}")
                    definitions = topic.get("definitions", [])
                    if definitions:
                        lines.append(f"")
                        for d in definitions[:3]:
                            lines.append(f"- **{cls._escape_md(d.get('term', ''))}**: {cls._escape_md(d.get('definition', ''))}")
                    examples = topic.get("examples", [])
                    if examples:
                        lines.append(f"")
                        lines.append(f"_Example: {cls._escape_md(examples[0])}_")
                    formulas = topic.get("formulas", [])
                    if formulas:
                        lines.append(f"")
                        for fm in formulas[:3]:
                            lines.append(f"- `{cls._escape_md(fm)}`")
                    emphasis = topic.get("emphasis", [])
                    if emphasis:
                        lines.append(f"")
                        for emp in emphasis[:2]:
                            src_label = "Lecturer Emphasis" if emp.get("source") == "lecturer_explicit" else "Key Emphasis"
                            lines.append(f"- **{src_label}:** {cls._escape_md(emp.get('text', ''))}")
                    lines.append(f"")

                # Master Study Guide (14 Sections) — High-Yield label permitted ONLY when quality_pass=True and is_fallback=False
                guide_md = notes.get("markdown_study_guide", "")
                if guide_md:
                    qg = cls._evaluate_study_guide_quality(data)
                    lines.append(f"---")
                    lines.append(f"")
                    if qg["is_high_yield"]:
                        lines.append(f"## High-Yield Master Study Guide (14 Sections)")
                    else:
                        lines.append(f"## Lecture Study Guide")
                        lines.append(f"")
                        lines.append(f"> **Notice (Fallback / Limited-Quality Output):** {qg['warning']}")
                        if qg["persisted_coverage_pct"] is not None:
                            lines.append(f"> - **Persisted transcript temporal coverage:** {qg['persisted_coverage_pct']:.2f}%")
                        if qg["high_trust_coverage_pct"] is not None:
                            below_note = " (below the 80% quality threshold)" if qg["high_trust_coverage_pct"] < 80.0 else ""
                            lines.append(f"> - **High-trust evidence coverage used by notes validation:** {qg['high_trust_coverage_pct']:.2f}%{below_note}")
                    lines.append(f"")
                    lines.append(guide_md.strip())
                    lines.append(f"")

            else:
                # Legacy fallback
                if notes.get("topics"):
                    lines.append(f"### Main Topics")
                    for top in notes["topics"]:
                        lines.append(f"- {cls._escape_md(top)}")
                    lines.append(f"")

                if notes.get("subtopics"):
                    lines.append(f"### Subtopics")
                    for sub in notes["subtopics"]:
                        lines.append(f"- {cls._escape_md(sub)}")
                    lines.append(f"")

                if notes.get("definitions"):
                    lines.append(f"### Key Definitions")
                    for d in notes["definitions"]:
                        term = cls._escape_md(d.get("term", "Term"))
                        definition = cls._escape_md(d.get("definition", "Definition"))
                        lines.append(f"- **{term}**: {definition}")
                    lines.append(f"")

                if notes.get("key_concepts"):
                    lines.append(f"### Key Concepts")
                    for kc in notes["key_concepts"]:
                        lines.append(f"- {cls._escape_md(kc)}")
                    lines.append(f"")

            if notes.get("important_points"):
                lines.append(f"### Important & Exam-Relevant Points")
                for pt in notes["important_points"]:
                    timestamp = pt.get("timestamp", "Not specified")
                    point_text = cls._escape_md(pt.get("point", ""))
                    exam_tag = " **[EXAM RELEVANT]**" if pt.get("is_exam_relevant") else ""
                    sig = f" (Signal: _{cls._escape_md(pt.get('signal_phrase'))}_)" if pt.get("signal_phrase") else ""
                    lines.append(f"- `{timestamp}` {point_text}{exam_tag}{sig}")
                lines.append(f"")

            lines.append(f"---")
            lines.append(f"")


        # 3. Assignments
        if content_type in ["full", "assignments"]:
            lines.append(f"## Assignments & Coursework")
            lines.append(f"")
            assignments = data["assignments"]
            if not assignments:
                lines.append(f"_No assignments extracted from this recording._")
                lines.append(f"")
            else:
                for idx, asgn in enumerate(assignments, 1):
                    lines.append(f"### Assignment {idx}: {cls._escape_md(asgn.get('topic', 'Not specified'))}")
                    lines.append(f"- **Topic:** {cls._escape_md(asgn.get('topic', 'Not specified'))}")
                    lines.append(f"- **Required Tasks:** {cls._escape_md(asgn.get('required_tasks', 'Not specified'))}")
                    lines.append(f"- **Format:** {cls._escape_md(asgn.get('format', 'Not specified'))}")
                    lines.append(f"- **Length Requirements:** {cls._escape_md(asgn.get('length_requirements', 'Not specified'))}")
                    lines.append(f"- **Submission Method:** {cls._escape_md(asgn.get('submission_method', 'Not specified'))}")
                    lines.append(f"- **Deadline:** {cls._escape_md(asgn.get('deadline', 'Not specified'))}")
                    lines.append(f"- **Type:** {cls._escape_md(asgn.get('individual_or_group', 'Not specified'))}")
                    lines.append(f"- **Required Tools:** {cls._escape_md(asgn.get('required_tools', 'Not specified'))}")
                    lines.append(f"- **Confidence:** {cls._escape_md(asgn.get('confidence', 'Not specified'))}")
                    lines.append(f"- **Overall Timestamp:** `{asgn.get('timestamp', 'Not specified')}`")
                    if asgn.get("source_timestamps"):
                        src_ts = ", ".join(f"`{t}`" for t in asgn["source_timestamps"])
                        lines.append(f"- **Source Timestamps:** {src_ts}")
                    if asgn.get("field_timestamps"):
                        fts_lines = [f"`{cls._escape_md(k)}`: `{cls._escape_md(v)}`" for k, v in asgn["field_timestamps"].items()]
                        lines.append(f"- **Field Timestamps:** {', '.join(fts_lines)}")
                    lines.append(f"")
            lines.append(f"---")
            lines.append(f"")

        # 4. Action Items
        if content_type in ["full", "action_items"]:
            lines.append(f"## Action Items & Meeting Deliverables")
            lines.append(f"")
            action_items = data["action_items"]
            if not action_items:
                lines.append(f"_No action items extracted from this recording._")
                lines.append(f"")
            else:
                for idx, act in enumerate(action_items, 1):
                    lines.append(f"### Action Item {idx}")
                    lines.append(f"- **Task:** {cls._escape_md(act.get('task', 'Not specified'))}")
                    lines.append(f"- **Owner:** {cls._escape_md(act.get('owner', 'Not specified'))}")
                    lines.append(f"- **Deadline:** {cls._escape_md(act.get('deadline', 'Not specified'))}")
                    lines.append(f"- **Deliverables:** {cls._escape_md(act.get('deliverables', 'Not specified'))}")
                    lines.append(f"- **Confidence:** {cls._escape_md(act.get('confidence', 'Not specified'))}")
                    lines.append(f"- **Overall Timestamp:** `{act.get('timestamp', 'Not specified')}`")
                    if act.get("source_timestamps"):
                        src_ts = ", ".join(f"`{t}`" for t in act["source_timestamps"])
                        lines.append(f"- **Source Timestamps:** {src_ts}")
                    if act.get("field_timestamps"):
                        fts_lines = [f"`{cls._escape_md(k)}`: `{cls._escape_md(v)}`" for k, v in act["field_timestamps"].items()]
                        lines.append(f"- **Field Timestamps:** {', '.join(fts_lines)}")
                    lines.append(f"")
            lines.append(f"---")
            lines.append(f"")

        # 5. Practice Quizzes & Conceptual MCQs
        if content_type in ["full", "quizzes"]:
            lines.append(f"## Practice Quizzes & Conceptual MCQs")
            lines.append(f"")
            quizzes = data["quizzes"]
            if not quizzes:
                lines.append(f"_No practice quizzes available._")
                lines.append(f"")
            else:
                letters = ["A", "B", "C", "D"]
                for idx, q in enumerate(quizzes, 1):
                    cat = q.get("category")
                    cat_hdr = f" [{cat}]" if cat else ""
                    lines.append(f"### {idx}.{cat_hdr} {cls._escape_md(q.get('question', ''))}")
                    lines.append(f"")
                    opts = q.get("options", [])
                    for o_idx, opt in enumerate(opts):
                        l = letters[o_idx] if o_idx < len(letters) else f"{o_idx+1}"
                        lines.append(f"- **{l}.** {cls._escape_md(opt)}")
                    lines.append(f"")
                    letter = q.get("correct_option_letter")
                    ans_label = f"**Answer:** {letter} ({cls._escape_md(q.get('correct_answer', 'Not specified'))})" if letter else f"**Answer:** {cls._escape_md(q.get('correct_answer', 'Not specified'))}"
                    lines.append(f"- {ans_label}")
                    if q.get("calculation"):
                        lines.append(f"- **Calculation:**")
                        for calc_line in q["calculation"].split("\n"):
                            lines.append(f"  `{cls._escape_md(calc_line)}`")
                    lines.append(f"- **Explanation:** {cls._escape_md(q.get('explanation', 'Not specified'))}")
                    lines.append(f"- **Concept:** {cls._escape_md(q.get('grounded_concept', 'Not specified'))}")
                    lines.append(f"")
            lines.append(f"---")
            lines.append(f"")

        # 6. Transcripts (Original & Normalized Roman Urdu)
        if content_type in ["full", "transcript"]:
            lines.append(f"## Complete Transcripts")
            lines.append(f"")
            segments = data.get("segments", [])

            lines.append(f"### Original Transcript")
            lines.append(f"")
            if segments:
                for seg in segments:
                    ts = f"[{seg.start_time:.2f}s - {seg.end_time:.2f}s]"
                    lines.append(f"`{ts}` {cls._escape_md(seg.text)}")
            elif data.get("transcript") and data["transcript"].original_transcript:
                lines.append(cls._escape_md(data["transcript"].original_transcript))
            else:
                lines.append(f"_Transcript not available._")
            lines.append(f"")

            lines.append(f"### Normalized Roman Urdu Transcript")
            lines.append(f"")
            if segments:
                for seg in segments:
                    ts = f"[{seg.start_time:.2f}s - {seg.end_time:.2f}s]"
                    norm = seg.normalized_text if seg.normalized_text else seg.text
                    lines.append(f"`{ts}` {cls._escape_md(norm)}")
            elif data.get("transcript") and data["transcript"].normalized_transcript:
                lines.append(cls._escape_md(data["transcript"].normalized_transcript))
            else:
                lines.append(f"_Normalized transcript not available._")
            lines.append(f"")

        return "\n".join(lines)

    # =========================================================================
    # 2. PLAIN TEXT BUILDER
    # =========================================================================
    @classmethod
    def _build_plain_text(cls, data: dict, content_type: str) -> str:
        rec = data["recording"]
        lines = []
        lines.append("=" * 70)
        lines.append("LECTURE & MEETING INTELLIGENCE REPORT")
        lines.append("=" * 70)
        lines.append(f"Recording: {rec.filename if rec else 'Unknown'}")
        lines.append(f"Duration:  {rec.duration_seconds if rec else 0.0:.2f} seconds")
        lines.append(f"Scope:     {content_type}")
        lines.append("=" * 70)
        lines.append("")

        if content_type in ["full", "summary"]:
            lines.append("-" * 70)
            lines.append("1. EXECUTIVE SUMMARY")
            lines.append("-" * 70)
            lines.append("[Short Summary]")
            lines.append(data["summary_short"])
            lines.append("")
            lines.append("[Detailed Summary]")
            lines.append(data["summary_detailed"])
            lines.append("")

        if content_type in ["full", "notes"]:
            notes = data["lecture_notes"]
            lines.append("-" * 70)
            lines.append("2. STRUCTURED LECTURE NOTES")
            lines.append("-" * 70)
            if notes.get("topics"):
                lines.append("[Main Topics]")
                for t in notes["topics"]:
                    lines.append(f"  * {t}")
                lines.append("")
            if notes.get("subtopics"):
                lines.append("[Subtopics]")
                for st in notes["subtopics"]:
                    lines.append(f"  * {st}")
                lines.append("")
            if notes.get("definitions"):
                lines.append("[Definitions]")
                for d in notes["definitions"]:
                    lines.append(f"  * {d.get('term', '')}: {d.get('definition', '')}")
                lines.append("")
            if notes.get("key_concepts"):
                lines.append("[Key Concepts]")
                for kc in notes["key_concepts"]:
                    lines.append(f"  * {kc}")
                lines.append("")
            if notes.get("important_points"):
                lines.append("[Important & Exam-Relevant Points]")
                for pt in notes["important_points"]:
                    ts = pt.get("timestamp", "Not specified")
                    ex = " [EXAM RELEVANT]" if pt.get("is_exam_relevant") else ""
                    sig = f" (Signal: {pt.get('signal_phrase')})" if pt.get("signal_phrase") else ""
                    lines.append(f"  * {ts} {pt.get('point', '')}{ex}{sig}")
                lines.append("")

        if content_type in ["full", "assignments"]:
            lines.append("-" * 70)
            lines.append("3. ASSIGNMENTS & COURSEWORK")
            lines.append("-" * 70)
            assignments = data["assignments"]
            if not assignments:
                lines.append("No assignments extracted.")
            else:
                for idx, asgn in enumerate(assignments, 1):
                    lines.append(f"Assignment {idx}: {asgn.get('topic', 'Not specified')}")
                    lines.append(f"  Topic:             {asgn.get('topic', 'Not specified')}")
                    lines.append(f"  Required Tasks:    {asgn.get('required_tasks', 'Not specified')}")
                    lines.append(f"  Format:            {asgn.get('format', 'Not specified')}")
                    lines.append(f"  Length Reqs:       {asgn.get('length_requirements', 'Not specified')}")
                    lines.append(f"  Submission Method: {asgn.get('submission_method', 'Not specified')}")
                    lines.append(f"  Deadline:          {asgn.get('deadline', 'Not specified')}")
                    lines.append(f"  Type:              {asgn.get('individual_or_group', 'Not specified')}")
                    lines.append(f"  Required Tools:    {asgn.get('required_tools', 'Not specified')}")
                    lines.append(f"  Confidence:        {asgn.get('confidence', 'Not specified')}")
                    lines.append(f"  Timestamp:         {asgn.get('timestamp', 'Not specified')}")
                    if asgn.get("source_timestamps"):
                        lines.append(f"  Source Timestamps: {', '.join(asgn['source_timestamps'])}")
                    if asgn.get("field_timestamps"):
                        fts = ", ".join(f"{k}={v}" for k, v in asgn["field_timestamps"].items())
                        lines.append(f"  Field Timestamps:  {fts}")
                    lines.append("")

        if content_type in ["full", "action_items"]:
            lines.append("-" * 70)
            lines.append("4. ACTION ITEMS & DELIVERABLES")
            lines.append("-" * 70)
            action_items = data["action_items"]
            if not action_items:
                lines.append("No action items extracted.")
            else:
                for idx, act in enumerate(action_items, 1):
                    lines.append(f"Action Item {idx}:")
                    lines.append(f"  Task:              {act.get('task', 'Not specified')}")
                    lines.append(f"  Owner:             {act.get('owner', 'Not specified')}")
                    lines.append(f"  Deadline:          {act.get('deadline', 'Not specified')}")
                    lines.append(f"  Deliverables:      {act.get('deliverables', 'Not specified')}")
                    lines.append(f"  Confidence:        {act.get('confidence', 'Not specified')}")
                    lines.append(f"  Timestamp:         {act.get('timestamp', 'Not specified')}")
                    if act.get("source_timestamps"):
                        lines.append(f"  Source Timestamps: {', '.join(act['source_timestamps'])}")
                    if act.get("field_timestamps"):
                        fts = ", ".join(f"{k}={v}" for k, v in act["field_timestamps"].items())
                        lines.append(f"  Field Timestamps:  {fts}")
                    lines.append("")

        if content_type in ["full", "quizzes"]:
            lines.append("-" * 70)
            lines.append("5. PRACTICE QUIZZES & CONCEPTUAL MCQS")
            lines.append("-" * 70)
            quizzes = data["quizzes"]
            if not quizzes:
                lines.append("No practice quizzes available.")
            else:
                letters = ["A", "B", "C", "D"]
                for idx, q in enumerate(quizzes, 1):
                    cat = q.get("category")
                    cat_str = f" [{cat}]" if cat else ""
                    lines.append(f"{idx}.{cat_str} {q.get('question', '')}")
                    for o_idx, opt in enumerate(q.get("options", [])):
                        l = letters[o_idx] if o_idx < len(letters) else f"{o_idx+1}"
                        lines.append(f"   {l}. {opt}")
                    letter = q.get("correct_option_letter")
                    ans_label = f"{letter} ({q.get('correct_answer', '')})" if letter else q.get('correct_answer', '')
                    lines.append(f"   Answer: {ans_label}")
                    if q.get("calculation"):
                        lines.append(f"   Calculation:\n   " + q["calculation"].replace("\n", "\n   "))
                    lines.append(f"   Explanation: {q.get('explanation', 'Not specified')}")
                    lines.append(f"   Concept:     {q.get('grounded_concept', 'Not specified')}")
                    lines.append("")

        if content_type in ["full", "transcript"]:
            lines.append("-" * 70)
            lines.append("6. COMPLETE TRANSCRIPTS")
            lines.append("-" * 70)
            segments = data.get("segments", [])
            lines.append("[Original Spoken Transcript]")
            if segments:
                for seg in segments:
                    lines.append(f"[{seg.start_time:.2f}s - {seg.end_time:.2f}s] {seg.text}")
            elif data.get("transcript"):
                lines.append(data["transcript"].original_transcript)
            lines.append("")
            lines.append("[Normalized Roman Urdu Transcript]")
            if segments:
                for seg in segments:
                    norm = seg.normalized_text if seg.normalized_text else seg.text
                    lines.append(f"[{seg.start_time:.2f}s - {seg.end_time:.2f}s] {norm}")
            elif data.get("transcript"):
                lines.append(data["transcript"].normalized_transcript)
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # 3. DOCX BUILDER
    # =========================================================================
    @classmethod
    def _build_docx(cls, data: dict, content_type: str) -> bytes:
        from docx import Document
        from docx.shared import Pt, Inches, RGBColor

        doc = Document()

        # Document Title
        title_p = doc.add_heading("Lecture & Meeting Intelligence Report", level=0)
        rec = data["recording"]

        meta_p = doc.add_paragraph()
        meta_p.add_run(f"Recording: ").bold = True
        meta_p.add_run(f"{rec.filename if rec else 'Unknown'} | ")
        meta_p.add_run(f"Duration: ").bold = True
        meta_p.add_run(f"{rec.duration_seconds if rec else 0.0:.2f}s | ")
        meta_p.add_run(f"Scope: ").bold = True
        meta_p.add_run(f"{content_type}")

        # Summary
        if content_type in ["full", "summary"]:
            doc.add_heading("1. Executive Summary", level=1)
            doc.add_heading("Short Summary", level=2)
            doc.add_paragraph(data["summary_short"])
            doc.add_heading("Detailed Summary", level=2)
            doc.add_paragraph(data["summary_detailed"])

        # Notes
        if content_type in ["full", "notes"]:
            notes = data["lecture_notes"]
            doc.add_heading("2. Structured Lecture Notes", level=1)
            if notes.get("topics"):
                doc.add_heading("Topics", level=2)
                for t in notes["topics"]:
                    doc.add_paragraph(t, style="List Bullet")
            if notes.get("subtopics"):
                doc.add_heading("Subtopics", level=2)
                for st in notes["subtopics"]:
                    doc.add_paragraph(st, style="List Bullet")
            if notes.get("definitions"):
                doc.add_heading("Definitions", level=2)
                for d in notes["definitions"]:
                    p = doc.add_paragraph(style="List Bullet")
                    p.add_run(f"{d.get('term', '')}: ").bold = True
                    p.add_run(d.get("definition", ""))
            if notes.get("key_concepts"):
                doc.add_heading("Key Concepts", level=2)
                for kc in notes["key_concepts"]:
                    doc.add_paragraph(kc, style="List Bullet")
            if notes.get("important_points"):
                doc.add_heading("Important & Exam-Relevant Points", level=2)
                for pt in notes["important_points"]:
                    p = doc.add_paragraph(style="List Bullet")
                    p.add_run(f"[{pt.get('timestamp', 'Not specified')}] ").bold = True
                    p.add_run(pt.get("point", ""))
                    if pt.get("is_exam_relevant"):
                        tag_run = p.add_run(" [EXAM RELEVANT]")
                        tag_run.bold = True
                        tag_run.font.color.rgb = RGBColor(180, 0, 0)
                    if pt.get("signal_phrase"):
                        p.add_run(f" (Signal: {pt.get('signal_phrase')})").italic = True

        # Assignments
        if content_type in ["full", "assignments"]:
            doc.add_heading("3. Assignments & Coursework", level=1)
            assignments = data["assignments"]
            if not assignments:
                doc.add_paragraph("No assignments extracted.")
            else:
                for idx, asgn in enumerate(assignments, 1):
                    doc.add_heading(f"Assignment {idx}: {asgn.get('topic', 'Not specified')}", level=2)
                    for k, label in [
                        ("topic", "Topic"),
                        ("required_tasks", "Required Tasks"),
                        ("format", "Format"),
                        ("length_requirements", "Length Requirements"),
                        ("submission_method", "Submission Method"),
                        ("deadline", "Deadline"),
                        ("individual_or_group", "Individual/Group"),
                        ("required_tools", "Required Tools"),
                        ("confidence", "Confidence"),
                        ("timestamp", "Timestamp"),
                    ]:
                        p = doc.add_paragraph()
                        p.add_run(f"{label}: ").bold = True
                        p.add_run(str(asgn.get(k, "Not specified")))

                    if asgn.get("source_timestamps"):
                        p = doc.add_paragraph()
                        p.add_run("Source Timestamps: ").bold = True
                        p.add_run(", ".join(asgn["source_timestamps"]))

                    if asgn.get("field_timestamps"):
                        p = doc.add_paragraph()
                        p.add_run("Field Timestamps: ").bold = True
                        p.add_run(str(asgn["field_timestamps"]))

        # Action Items
        if content_type in ["full", "action_items"]:
            doc.add_heading("4. Action Items & Deliverables", level=1)
            action_items = data["action_items"]
            if not action_items:
                doc.add_paragraph("No action items extracted.")
            else:
                for idx, act in enumerate(action_items, 1):
                    doc.add_heading(f"Action Item {idx}", level=2)
                    for k, label in [
                        ("task", "Task"),
                        ("owner", "Owner"),
                        ("deadline", "Deadline"),
                        ("deliverables", "Deliverables"),
                        ("confidence", "Confidence"),
                        ("timestamp", "Timestamp"),
                    ]:
                        p = doc.add_paragraph()
                        p.add_run(f"{label}: ").bold = True
                        p.add_run(str(act.get(k, "Not specified")))

                    if act.get("source_timestamps"):
                        p = doc.add_paragraph()
                        p.add_run("Source Timestamps: ").bold = True
                        p.add_run(", ".join(act["source_timestamps"]))

                    if act.get("field_timestamps"):
                        p = doc.add_paragraph()
                        p.add_run("Field Timestamps: ").bold = True
                        p.add_run(str(act["field_timestamps"]))

        # Practice Quizzes
        if content_type in ["full", "quizzes"]:
            doc.add_heading("5. Practice Quizzes & Conceptual MCQs", level=1)
            quizzes = data["quizzes"]
            if not quizzes:
                doc.add_paragraph("No practice quizzes available.")
            else:
                letters = ["A", "B", "C", "D"]
                for idx, q in enumerate(quizzes, 1):
                    cat = q.get("category")
                    cat_str = f" [{cat}]" if cat else ""
                    doc.add_heading(f"{idx}.{cat_str} {q.get('question', '')}", level=2)
                    for o_idx, opt in enumerate(q.get("options", [])):
                        l = letters[o_idx] if o_idx < len(letters) else f"{o_idx+1}"
                        p_opt = doc.add_paragraph(style="List Bullet")
                        p_opt.add_run(f"{l}. ").bold = True
                        p_opt.add_run(opt)
                    p = doc.add_paragraph()
                    p.add_run("Answer: ").bold = True
                    letter = q.get("correct_option_letter")
                    ans_val = f"{letter} ({q.get('correct_answer', 'Not specified')})" if letter else q.get('correct_answer', 'Not specified')
                    p.add_run(ans_val)
                    if q.get("calculation"):
                        p_c = doc.add_paragraph()
                        p_c.add_run("Calculation:\n").bold = True
                        p_c.add_run(q["calculation"])
                    p2 = doc.add_paragraph()
                    p2.add_run("Explanation: ").bold = True
                    p2.add_run(q.get("explanation", "Not specified"))

        # Transcripts
        if content_type in ["full", "transcript"]:
            doc.add_heading("6. Complete Transcripts", level=1)
            segments = data.get("segments", [])
            doc.add_heading("Original Transcript", level=2)
            if segments:
                for seg in segments:
                    p = doc.add_paragraph()
                    p.add_run(f"[{seg.start_time:.2f}s - {seg.end_time:.2f}s] ").bold = True
                    p.add_run(seg.text)
            elif data.get("transcript"):
                doc.add_paragraph(data["transcript"].original_transcript)

            doc.add_heading("Normalized Roman Urdu Transcript", level=2)
            if segments:
                for seg in segments:
                    p = doc.add_paragraph()
                    p.add_run(f"[{seg.start_time:.2f}s - {seg.end_time:.2f}s] ").bold = True
                    norm = seg.normalized_text if seg.normalized_text else seg.text
                    p.add_run(norm)
            elif data.get("transcript"):
                doc.add_paragraph(data["transcript"].normalized_transcript)

        bio = io.BytesIO()
        doc.save(bio)
        return bio.getvalue()

    # =========================================================================
    # 4. PDF BUILDER
    # =========================================================================
    @classmethod
    def _build_pdf(cls, data: dict, content_type: str) -> bytes:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether

        bio = io.BytesIO()
        doc = SimpleDocTemplate(
            bio,
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("ExportTitle", parent=styles["Heading1"], fontSize=18, leading=22, textColor=colors.HexColor("#1A365D"))
        h1_style = ParagraphStyle("ExportH1", parent=styles["Heading2"], fontSize=14, leading=18, textColor=colors.HexColor("#2B6CB0"), spaceBefore=12, spaceAfter=6)
        h2_style = ParagraphStyle("ExportH2", parent=styles["Heading3"], fontSize=11, leading=14, textColor=colors.HexColor("#2D3748"), spaceBefore=8, spaceAfter=4)
        body_style = ParagraphStyle("ExportBody", parent=styles["Normal"], fontSize=9, leading=12, textColor=colors.HexColor("#2D3748"))
        bullet_style = ParagraphStyle("ExportBullet", parent=body_style, leftIndent=12, firstLineIndent=-8, spaceAfter=2)
        meta_style = ParagraphStyle("ExportMeta", parent=body_style, fontSize=8, textColor=colors.HexColor("#718096"))

        story = []

        def _esc(val: any) -> str:
            return html.escape(str(val)) if val is not None else ""

        # Title
        story.append(Paragraph("Lecture & Meeting Intelligence Report", title_style))
        rec = data["recording"]
        meta_text = (
            f"<b>Recording:</b> {_esc(rec.filename if rec else 'Unknown')} | "
            f"<b>Duration:</b> {rec.duration_seconds if rec else 0.0:.2f}s | "
            f"<b>Scope:</b> {_esc(content_type)}"
        )
        story.append(Paragraph(meta_text, meta_style))
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E0"), spaceAfter=10))

        # Summary
        if content_type in ["full", "summary"]:
            story.append(Paragraph("1. Executive Summary", h1_style))
            story.append(Paragraph("Short Summary", h2_style))
            story.append(Paragraph(_esc(data["summary_short"]), body_style))
            story.append(Spacer(1, 6))
            story.append(Paragraph("Detailed Summary", h2_style))
            story.append(Paragraph(_esc(data["summary_detailed"]), body_style))
            story.append(Spacer(1, 10))

        # Notes / Study Guide
        if content_type in ["full", "notes"]:
            notes = data["lecture_notes"]
            qg = cls._evaluate_study_guide_quality(data)
            if content_type == "notes":
                guide_title = "High-Yield Master Study Guide & Notes" if qg["is_high_yield"] else "Lecture Study Guide & Notes"
            else:
                guide_title = "2. Structured Lecture Notes"
            story.append(Paragraph(guide_title, h1_style))
            if not qg["is_high_yield"] and (notes.get("study_topics") or notes.get("markdown_study_guide")):
                cov_parts = []
                if qg["persisted_coverage_pct"] is not None:
                    cov_parts.append(f"Persisted transcript temporal coverage: {qg['persisted_coverage_pct']:.2f}%.")
                if qg["high_trust_coverage_pct"] is not None:
                    below_note = " (below the 80% quality threshold)" if qg["high_trust_coverage_pct"] < 80.0 else ""
                    cov_parts.append(f"High-trust evidence coverage used by notes validation: {qg['high_trust_coverage_pct']:.2f}%{below_note}.")
                notice_text = f"<b>Fallback / Limited-Quality Output:</b> {_esc(qg['warning'])}"
                if cov_parts:
                    notice_text += " " + _esc(" ".join(cov_parts))
                story.append(Paragraph(notice_text, body_style))
                story.append(Spacer(1, 6))

            study_topics = notes.get("study_topics", [])
            overview = notes.get("lecture_overview", "")
            guide_md = notes.get("markdown_study_guide", "")

            # Render Study Topics if present
            if study_topics:
                if overview:
                    story.append(Paragraph(f"<b>Overview:</b> {_esc(overview)}", body_style))
                    story.append(Spacer(1, 4))

                for i, topic in enumerate(study_topics, 1):
                    t_title = _esc(topic.get("title", "Topic"))
                    spans = topic.get("spans", [])
                    ts_str = f" [{spans[0].get('start',0):.0f}s – {spans[-1].get('end',0):.0f}s]" if spans else ""
                    exam_tag = " <font color='#C53030'><b>[EXAM RELEVANT]</b></font>" if topic.get("exam_relevant") else ""
                    story.append(Paragraph(f"{i}. {t_title}{exam_tag} <font color='#4F46E5'>{ts_str}</font>", h2_style))
                    
                    if topic.get("explanation"):
                        story.append(Paragraph(_esc(topic["explanation"]), body_style))
                    
                    for kp in topic.get("key_points", [])[:6]:
                        story.append(Paragraph(f"• {_esc(kp)}", bullet_style))
                    
                    for d in topic.get("definitions", [])[:3]:
                        story.append(Paragraph(f"• <b>{_esc(d.get('term',''))}:</b> {_esc(d.get('definition',''))}", bullet_style))
                    
                    if topic.get("examples"):
                        story.append(Paragraph(f"<i>Example: {_esc(topic['examples'][0])}</i>", body_style))
                    
                    if topic.get("formulas"):
                        for fm in topic["formulas"][:3]:
                            story.append(Paragraph(f"<font face='Courier'><b>{_esc(fm)}</b></font>", bullet_style))
                    
                    story.append(Spacer(1, 4))

            # Render legacy fields (including important_points with [EXAM RELEVANT])
            if notes.get("topics") and not study_topics:
                story.append(Paragraph("Topics", h2_style))
                for t in notes["topics"]:
                    story.append(Paragraph(f"• {_esc(t)}", bullet_style))
            if notes.get("subtopics") and not study_topics:
                story.append(Paragraph("Subtopics", h2_style))
                for st in notes["subtopics"]:
                    story.append(Paragraph(f"• {_esc(st)}", bullet_style))
            if notes.get("definitions") and not study_topics:
                story.append(Paragraph("Definitions", h2_style))
                for d in notes["definitions"]:
                    story.append(Paragraph(f"• <b>{_esc(d.get('term', ''))}:</b> {_esc(d.get('definition', ''))}", bullet_style))
            if notes.get("key_concepts") and not study_topics:
                story.append(Paragraph("Key Concepts", h2_style))
                for kc in notes["key_concepts"]:
                    story.append(Paragraph(f"• {_esc(kc)}", bullet_style))
            if notes.get("important_points"):
                story.append(Paragraph("Important & Exam-Relevant Points", h2_style))
                for pt in notes["important_points"]:
                    ex = " <font color='#C53030'><b>[EXAM RELEVANT]</b></font>" if pt.get("is_exam_relevant") else ""
                    sig = f" <i>(Signal: {_esc(pt.get('signal_phrase'))})</i>" if pt.get("signal_phrase") else ""
                    story.append(Paragraph(f"• <b>[{_esc(pt.get('timestamp', 'Not specified'))}]</b> {_esc(pt.get('point', ''))}{ex}{sig}", bullet_style))

            # If 14-Section Markdown Study Guide is present, render it with rich formatting
            if guide_md:
                story.append(Spacer(1, 6))
                story.append(Paragraph("Comprehensive 14-Section Pedagogical Guide", h2_style))
                guide_lines = guide_md.splitlines()
                g_idx = 0
                while g_idx < len(guide_lines):
                    g_line = guide_lines[g_idx].strip()
                    if not g_line:
                        g_idx += 1
                        continue

                    if g_line.startswith("# "):
                        story.append(Paragraph(_esc(g_line[2:].strip()), h1_style))
                    elif g_line.startswith("## "):
                        story.append(Paragraph(_esc(g_line[3:].strip()), h1_style))
                    elif g_line.startswith("### "):
                        story.append(Paragraph(_esc(g_line[4:].strip()), h2_style))
                    elif g_line.startswith("#### "):
                        story.append(Paragraph(_esc(g_line[5:].strip()), ParagraphStyle("H4_G", parent=body_style, fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=2)))
                    elif g_line.startswith("---") or g_line.startswith("***"):
                        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0"), spaceAfter=6, spaceBefore=6))
                    elif g_line.startswith("- ") or g_line.startswith("* ") or g_line.startswith("• "):
                        txt = g_line[2:].strip()
                        txt = _esc(txt)
                        txt = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', txt)
                        txt = re.sub(r'\*(.*?)\*', r'<i>\1</i>', txt)
                        txt = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', txt)
                        story.append(Paragraph(f"• {txt}", bullet_style))
                    elif g_line.startswith("$$") and g_line.endswith("$$") and len(g_line) > 4:
                        f_txt = _esc(g_line[2:-2].strip())
                        story.append(Paragraph(f"<b>{f_txt}</b>", ParagraphStyle("Fm_G", parent=body_style, fontName="Courier-Bold", backColor=colors.HexColor("#F8FAFC"), borderPadding=4, spaceBefore=4, spaceAfter=4)))
                    elif g_line.startswith("|") and g_line.endswith("|"):
                        table_lines = []
                        while g_idx < len(guide_lines) and guide_lines[g_idx].strip().startswith("|") and guide_lines[g_idx].strip().endswith("|"):
                            table_lines.append(guide_lines[g_idx].strip())
                            g_idx += 1
                        g_idx -= 1
                        
                        table_data = []
                        for tl in table_lines:
                            if "---" in tl:
                                continue
                            cells = [c.strip() for c in tl.split("|")[1:-1]]
                            row_cells = []
                            for c in cells:
                                esc_c = _esc(c)
                                esc_c = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', esc_c)
                                row_cells.append(Paragraph(esc_c, body_style))
                            table_data.append(row_cells)
                        
                        if table_data:
                            col_count = max(len(r) for r in table_data)
                            if col_count > 0:
                                col_w = 540 / col_count
                                t = Table(table_data, colWidths=[col_w]*col_count)
                                t.setStyle(TableStyle([
                                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
                                    ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#0F172A")),
                                    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                                    ('TOPPADDING', (0,0), (-1,-1), 4),
                                    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
                                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                                ]))
                                story.append(Spacer(1, 4))
                                story.append(t)
                                story.append(Spacer(1, 4))
                    else:
                        txt = _esc(g_line)
                        txt = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', txt)
                        txt = re.sub(r'\*(.*?)\*', r'<i>\1</i>', txt)
                        txt = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', txt)
                        story.append(Paragraph(txt, body_style))
                    g_idx += 1

            story.append(Spacer(1, 10))

        # Assignments
        if content_type in ["full", "assignments"]:
            story.append(Paragraph("3. Assignments & Coursework", h1_style))
            assignments = data["assignments"]
            if not assignments:
                story.append(Paragraph("No assignments extracted.", body_style))
            else:
                for idx, asgn in enumerate(assignments, 1):
                    asgn_block = []
                    asgn_block.append(Paragraph(f"Assignment {idx}: {_esc(asgn.get('topic', 'Not specified'))}", h2_style))
                    for k, label in [
                        ("topic", "Topic"),
                        ("required_tasks", "Required Tasks"),
                        ("format", "Format"),
                        ("length_requirements", "Length Requirements"),
                        ("submission_method", "Submission Method"),
                        ("deadline", "Deadline"),
                        ("individual_or_group", "Individual/Group"),
                        ("required_tools", "Required Tools"),
                        ("confidence", "Confidence"),
                        ("timestamp", "Timestamp"),
                    ]:
                        val = str(asgn.get(k, "Not specified"))
                        asgn_block.append(Paragraph(f"<b>{_esc(label)}:</b> {_esc(val)}", body_style))

                    if asgn.get("source_timestamps"):
                        src_str = ", ".join(asgn["source_timestamps"])
                        asgn_block.append(Paragraph(f"<b>Source Timestamps:</b> {_esc(src_str)}", body_style))
                    if asgn.get("field_timestamps"):
                        fts_str = ", ".join(f"{k}={v}" for k, v in asgn["field_timestamps"].items())
                        asgn_block.append(Paragraph(f"<b>Field Timestamps:</b> {_esc(fts_str)}", body_style))
                    asgn_block.append(Spacer(1, 6))
                    story.append(KeepTogether(asgn_block))
            story.append(Spacer(1, 10))

        # Action Items
        if content_type in ["full", "action_items"]:
            story.append(Paragraph("4. Action Items & Deliverables", h1_style))
            action_items = data["action_items"]
            if not action_items:
                story.append(Paragraph("No action items extracted.", body_style))
            else:
                for idx, act in enumerate(action_items, 1):
                    act_block = []
                    act_block.append(Paragraph(f"Action Item {idx}", h2_style))
                    for k, label in [
                        ("task", "Task"),
                        ("owner", "Owner"),
                        ("deadline", "Deadline"),
                        ("deliverables", "Deliverables"),
                        ("confidence", "Confidence"),
                        ("timestamp", "Timestamp"),
                    ]:
                        val = str(act.get(k, "Not specified"))
                        act_block.append(Paragraph(f"<b>{_esc(label)}:</b> {_esc(val)}", body_style))

                    if act.get("source_timestamps"):
                        src_str = ", ".join(act["source_timestamps"])
                        act_block.append(Paragraph(f"<b>Source Timestamps:</b> {_esc(src_str)}", body_style))
                    if act.get("field_timestamps"):
                        fts_str = ", ".join(f"{k}={v}" for k, v in act["field_timestamps"].items())
                        act_block.append(Paragraph(f"<b>Field Timestamps:</b> {_esc(fts_str)}", body_style))
                    act_block.append(Spacer(1, 6))
                    story.append(KeepTogether(act_block))
            story.append(Spacer(1, 10))

        # Practice Quizzes
        if content_type in ["full", "quizzes"]:
            story.append(Paragraph("5. Practice Quizzes & Conceptual MCQs", h1_style))
            quizzes = data["quizzes"]
            if not quizzes:
                story.append(Paragraph("No practice quizzes available.", body_style))
            else:
                letters = ["A", "B", "C", "D"]
                for idx, q in enumerate(quizzes, 1):
                    q_block = []
                    cat = q.get("category")
                    cat_str = f" [{cat}]" if cat else ""
                    q_block.append(Paragraph(f"{idx}.{cat_str} {_esc(q.get('question', ''))}", h2_style))
                    for o_idx, opt in enumerate(q.get("options", [])):
                        l = letters[o_idx] if o_idx < len(letters) else f"{o_idx+1}"
                        q_block.append(Paragraph(f"<b>{l}.</b> {_esc(opt)}", bullet_style))
                    letter = q.get("correct_option_letter")
                    ans_val = f"{letter} ({_esc(q.get('correct_answer', 'Not specified'))})" if letter else _esc(q.get('correct_answer', 'Not specified'))
                    q_block.append(Paragraph(f"<b>Answer:</b> {ans_val}", body_style))
                    if q.get("calculation"):
                        calc_escaped = _esc(q["calculation"]).replace("\n", "<br/>")
                        q_block.append(Paragraph(f"<b>Calculation:</b><br/><font face='Courier'>{calc_escaped}</font>", body_style))
                    q_block.append(Paragraph(f"<b>Explanation:</b> {_esc(q.get('explanation', 'Not specified'))}", body_style))
                    q_block.append(Spacer(1, 6))
                    story.append(KeepTogether(q_block))
            story.append(Spacer(1, 10))

        # Complete Transcripts
        if content_type in ["full", "transcript"]:
            story.append(Paragraph("6. Complete Transcripts", h1_style))
            segments = data.get("segments", [])

            story.append(Paragraph("Original Transcript", h2_style))
            if segments:
                for seg in segments:
                    ts = f"[{seg.start_time:.2f}s - {seg.end_time:.2f}s]"
                    story.append(Paragraph(f"<b>{_esc(ts)}</b> {_esc(seg.text)}", body_style))
            elif data.get("transcript"):
                story.append(Paragraph(_esc(data["transcript"].original_transcript), body_style))

            story.append(Spacer(1, 6))
            story.append(Paragraph("Normalized Roman Urdu Transcript", h2_style))
            if segments:
                for seg in segments:
                    ts = f"[{seg.start_time:.2f}s - {seg.end_time:.2f}s]"
                    norm = seg.normalized_text if seg.normalized_text else seg.text
                    story.append(Paragraph(f"<b>{_esc(ts)}</b> {_esc(norm)}", body_style))
            elif data.get("transcript"):
                story.append(Paragraph(_esc(data["transcript"].normalized_transcript), body_style))

        doc.build(story)
        return bio.getvalue()
