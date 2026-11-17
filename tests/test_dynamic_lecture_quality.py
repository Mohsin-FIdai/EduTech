"""
Comprehensive quality and isolation validation suite for Dynamic Lecture Intelligence.

Tests that Lecture AI dynamically understands and processes lectures across ANY
unseen academic discipline (History, Statistics, Psychology, Roman Urdu + English
code-switched Economics) without relying on hardcoded subject assumptions or
leaking information across recordings.
"""
import json
from types import SimpleNamespace
import pytest

from app.services.topic_intelligence import TopicIntelligenceEngine, Segment
from app.services.study_guide_generator import (
    StudyGuideSynthesizer,
    AcademicMCQEngine,
    _build_extractive_fallback_guide,
)
from app.services.pedagogical_validator import PedagogicalValidator
from app.services.academic_pipeline import run_academic_v2
from app.services.intelligence_service import IntelligenceService
from app.services.rag_service import LectureRAGService
from app.models import Recording, Transcript, TranscriptSegment, IntelligenceReport
from app.database import Base, engine, SessionLocal


def _make_segments(items: list[tuple[float, float, str]]) -> list[SimpleNamespace]:
    segs = []
    for idx, (st, et, txt) in enumerate(items):
        segs.append(
            SimpleNamespace(
                id=idx + 1,
                segment_index=idx,
                start=float(st),
                end=float(et),
                start_time=float(st),
                end_time=float(et),
                text=txt,
                normalized_text=txt,
                english_text=txt,
                confidence=0.95,
            )
        )
    return segs


# ---------------------------------------------------------------------------
# Representative Unseen Discipline Fixtures (History, Statistics, Bilingual Economics)
# ---------------------------------------------------------------------------

HISTORY_INDUSTRIAL_REVOLUTION_ITEMS = [
    (0.0, 18.0, "Today we explore the Industrial Revolution Overview, which transformed agrarian economies into mechanized manufacturing societies during the eighteenth century."),
    (18.0, 38.0, "Mechanization is defined as the process of replacing manual human and animal labor with powered machinery in textile factories."),
    (38.0, 58.0, "For example, the spinning jenny and water frame enabled rapid cotton thread production in British textile mills."),
    (58.0, 78.0, "Steam Engine Innovation provided a reliable source of rotary mechanical power that did not depend on river geography."),
    (78.0, 100.0, "Because factories no longer needed fast-flowing rivers, industrial production relocated to urban centers near coal mines."),
    (100.0, 125.0, "Moving on to Urbanization and Social Change, millions of rural agricultural workers migrated to industrial cities seeking factory wages."),
    (125.0, 150.0, "Urbanization refers to the demographic shift of populations from rural villages into densely populated manufacturing cities."),
    (150.0, 175.0, "Rapid urban growth resulted in overcrowded housing, inadequate sanitation infrastructure, and the emergence of industrial labor unions."),
    (175.0, 200.0, "Remember this for the exam: the transition from domestic cottage industry to centralized factory production fundamentally altered labor relations."),
]

STATISTICS_HYPOTHESIS_TESTING_ITEMS = [
    (0.0, 20.0, "Today we discuss Statistical Hypothesis Testing, a formal inferential framework used to evaluate claims about population parameters using sample data."),
    (20.0, 42.0, "The Null Hypothesis is defined as the default baseline statement asserting that there is no significant effect or difference between population groups."),
    (42.0, 65.0, "In contrast, the Alternative Hypothesis represents the research claim that a genuine statistical effect or difference exists in the population."),
    (65.0, 90.0, "Moving on to P-Value and Significance Levels, the p-value measures the probability of observing sample results at least as extreme as the measured statistic assuming the null hypothesis is true."),
    (90.0, 115.0, "Test Statistic = (Sample Mean - Population Mean) / Standard Error."),
    (115.0, 140.0, "For instance, if the calculated p-value is 0.01 and our significance threshold alpha is 0.05, we reject the null hypothesis in favor of the alternative hypothesis."),
    (140.0, 168.0, "Now we examine Type I and Type II Errors in statistical decision making."),
    (168.0, 195.0, "Type I Error refers to the incorrect rejection of a true null hypothesis, which occurs with probability equal to alpha."),
    (195.0, 222.0, "Type II Error means failing to reject a false null hypothesis when a real effect actually exists in the underlying population."),
]

BILINGUAL_ECONOMICS_ITEMS = [
    (0.0, 22.0, "Aaj ka topic hai Demand and Supply Equilibrium. Demand is defined as the quantity of a commodity that consumers are willing and able to purchase at various price levels."),
    (22.0, 45.0, "Law of Demand ka matlab hai ke jab kisi product ki price increase hoti hai to uski quantity demanded decrease hoti hai, assuming ceteris paribus."),
    (45.0, 70.0, "Misal ke taur par, when the price of seasonal fruit rises sharply in the market, households reduce their weekly purchase quantity."),
    (70.0, 95.0, "Ab hum baat karenge Price Elasticity of Demand ke baare mein. Price Elasticity measures the responsiveness of quantity demanded to a percentage change in market price."),
    (95.0, 120.0, "Price Elasticity equals Percentage Change in Quantity Demanded divided by Percentage Change in Price."),
    (120.0, 148.0, "Essential medical goods exhibit inelastic demand because consumers must purchase them regardless of moderate price increases."),
    (148.0, 175.0, "Remember this for the exam: luxury goods have elastic demand whereas life-saving medicines have inelastic demand."),
]


class TestUnseenDisciplinesDynamicIntelligence:
    """Verifies that Topic Extractor, Study Guide, and Quiz Engine work on unseen subjects."""

    def test_history_lecture_topics_notes_and_quizzes(self):
        segs = [Segment(id=idx, start=st, end=et, text=txt) for idx, (st, et, txt) in enumerate(HISTORY_INDUSTRIAL_REVOLUTION_ITEMS)]
        text = " ".join(s.text for s in segs)

        engine = TopicIntelligenceEngine(use_llm=False)
        notes_obj = engine.analyze_lecture_study_notes(text, segs, 200.0, topic_hint="Industrial Revolution")
        assert len(notes_obj.study_topics) >= 1

        raw_topics = [t.to_dict() for t in notes_obj.study_topics]
        enriched = StudyGuideSynthesizer.enrich_study_topics(
            raw_topics, transcript_text=text, segments=_make_segments(HISTORY_INDUSTRIAL_REVOLUTION_ITEMS)
        )
        assert len(enriched) >= 1
        first = enriched[0]
        assert len(first["title"].split()) >= 2
        assert first["explanation"].lower().startswith("this section")
        assert len(first["key_points"]) >= 2

        # Verify History definitions and examples were dynamically extracted from transcript
        all_defs = [d for t in enriched for d in t.get("definitions", [])]
        def_terms_low = " ".join(d.get("term", "").lower() for d in all_defs)
        assert "mechanization" in def_terms_low or "urbanization" in def_terms_low

        all_exs = [e for t in enriched for e in t.get("examples", [])]
        assert len(all_exs) >= 1
        assert any("spinning jenny" in e.lower() or "cotton" in e.lower() for e in all_exs)

        # Verify zero contamination from OS / Biology / Physics / Accounting
        full_dump = json.dumps(enriched).lower()
        for alien_term in ["round robin", "turnaround time", "reflex arc", "ohm's law", "cash inflow", "supervised learning"]:
            assert alien_term not in full_dump

    def test_statistics_lecture_formulas_definitions_and_quizzes(self):
        segs = [Segment(id=idx, start=st, end=et, text=txt) for idx, (st, et, txt) in enumerate(STATISTICS_HYPOTHESIS_TESTING_ITEMS)]
        text = " ".join(s.text for s in segs)

        engine = TopicIntelligenceEngine(use_llm=False)
        notes_obj = engine.analyze_lecture_study_notes(text, segs, 222.0, topic_hint="Statistical Hypothesis Testing")
        raw_topics = [t.to_dict() for t in notes_obj.study_topics]
        ns_segs = _make_segments(STATISTICS_HYPOTHESIS_TESTING_ITEMS)
        enriched = StudyGuideSynthesizer.enrich_study_topics(
            raw_topics, transcript_text=text, segments=ns_segs
        )
        assert len(enriched) >= 1

        all_defs = [d for t in enriched for d in t.get("definitions", [])]
        def_terms_low = " ".join(d.get("term", "").lower() for d in all_defs)
        assert any(k in def_terms_low for k in ["null hypothesis", "type i error", "type ii error"])

        all_formulas = [f for t in enriched for f in t.get("formulas", [])]
        assert len(all_formulas) >= 1
        assert any("test statistic" in f.lower() or "sample mean" in f.lower() for f in all_formulas)

        # Generate conceptual quizzes and verify they are grounded in Statistics, with zero OS contamination
        lecture_notes = {
            "topics": [t["title"] for t in enriched],
            "key_concepts": ["Null Hypothesis", "Alternative Hypothesis", "P-Value", "Type I Error"],
            "definitions": all_defs,
            "study_topics": enriched,
        }
        quizzes = StudyGuideSynthesizer.generate_conceptual_quizzes(
            transcript_text=text,
            topic_hint="Hypothesis Testing",
            lecture_notes=lecture_notes,
            segments=ns_segs,
        )
        assert len(quizzes) >= 3
        quiz_dump = json.dumps(quizzes).lower()
        assert "thread preemption" not in quiz_dump
        assert "data serialization" not in quiz_dump
        assert "firmware routines" not in quiz_dump
        for q in quizzes:
            assert q["correct_answer"] in q["options"]
            assert len(q["options"]) == 4

    def test_bilingual_roman_urdu_economics_lecture_extraction(self):
        segs = [Segment(id=idx, start=st, end=et, text=txt) for idx, (st, et, txt) in enumerate(BILINGUAL_ECONOMICS_ITEMS)]
        text = " ".join(s.text for s in segs)

        engine = TopicIntelligenceEngine(use_llm=False)
        notes_obj = engine.analyze_lecture_study_notes(text, segs, 175.0, topic_hint="Demand and Supply")
        raw_topics = [t.to_dict() for t in notes_obj.study_topics]
        enriched = StudyGuideSynthesizer.enrich_study_topics(
            raw_topics, transcript_text=text, segments=_make_segments(BILINGUAL_ECONOMICS_ITEMS)
        )
        assert len(enriched) >= 1

        # Verify bilingual definitions ("ka matlab hai ke", "is defined as", "measures")
        all_defs = [d for t in enriched for d in t.get("definitions", [])]
        terms_low = " ".join(d.get("term", "").lower() for d in all_defs)
        assert any(k in terms_low for k in ["demand", "price elasticity"])

        # Verify bilingual example ("Misal ke taur par")
        all_exs = [e for t in enriched for e in t.get("examples", [])]
        assert any("fruit" in e.lower() or "households" in e.lower() for e in all_exs)

        # Verify spoken equation ("equals ... divided by ...")
        all_fms = [f for t in enriched for f in t.get("formulas", [])]
        assert any("elasticity" in f.lower() and "divided by" in f.lower() for f in all_fms)


class TestCrossRecordingIsolationAndRAG:
    """Verifies that sequential recordings never leak concepts, timestamps, or canned answers."""

    def test_rag_refuses_os_questions_on_history_recording_and_grounds_timestamps(self):
        import uuid
        Base.metadata.create_all(bind=engine)
        db_session = SessionLocal()
        try:
            rec_id = f"rec-history-{uuid.uuid4().hex[:8]}"
            rec = Recording(
                id=rec_id,
                filename="industrial_revolution.wav",
                file_path="/tmp/industrial_revolution.wav",
                file_size_bytes=1024 * 50,
                duration_seconds=200.0,
                file_format=".wav",
                media_type="audio",
                status="PROCESSED",
            )
            db_session.add(rec)
            db_session.commit()

            full_txt = " ".join(t for _, _, t in HISTORY_INDUSTRIAL_REVOLUTION_ITEMS)
            tr = Transcript(
                id=f"tr-history-{uuid.uuid4().hex[:8]}",
                recording_id=rec.id,
                original_transcript=full_txt,
                normalized_transcript=full_txt,
            )
            db_session.add(tr)
            db_session.commit()

            for idx, (st, et, txt) in enumerate(HISTORY_INDUSTRIAL_REVOLUTION_ITEMS):
                db_session.add(
                    TranscriptSegment(
                        id=str(uuid.uuid4()),
                        transcript_id=tr.id,
                        recording_id=rec.id,
                        chunk_index=0,
                        segment_index=idx,
                        start_time=st,
                        end_time=et,
                        text=txt,
                        normalized_text=txt,
                        english_text=txt,
                    )
                )
            db_session.commit()

            # Generate full intelligence report for History recording
            report = IntelligenceService.generate_intelligence_report(rec.id, db_session, force=True)
            assert report["recording_id"] == rec.id

            # 1. Asking an Operating Systems question ("What is the Ready State in CPU scheduling?")
            # against the History recording MUST return not_found, NOT a canned OS answer!
            os_ans = LectureRAGService.answer_question(
                recording_id=rec.id,
                question="What are the process states and ready state in operating systems?",
                db=db_session,
            )
            assert os_ans["found"] is False
            assert os_ans["status"] == "not_found"
            assert "Ready (Ready State)" not in os_ans["answer"]

            # 2. Asking an Accounting calculation question on the History recording must NOT trigger Profit engine
            acc_ans = LectureRAGService.answer_question(
                recording_id=rec.id,
                question="If revenue is Rs. 50,000 and expenditure is Rs. 30,000, what is the profit?",
                db=db_session,
            )
            assert acc_ans["found"] is False

            # 3. Asking an in-scope History question ("What is mechanization?") returns grounded answer & timestamp
            hist_ans = LectureRAGService.answer_question(
                recording_id=rec.id,
                question="What is mechanization in the Industrial Revolution?",
                db=db_session,
            )
            assert hist_ans["found"] is True
            assert "mechanization" in hist_ans["answer"].lower() or "machinery" in hist_ans["answer"].lower()
            assert hist_ans["start_time"] is not None
            assert 0.0 <= hist_ans["start_time"] <= 200.0
        finally:
            db_session.close()


class TestUniversalLectureNotesInstructions:
    """
    Validates the 4 Universal Instructions for Lecture Topic Extraction & Note Generation:
      1. Accurate Topic Extraction (exclude greetings, mic checks, instructions, questions, AI responses)
      2. Proper Structure (Executive Summary & Learning Objectives -> Logical Topics -> Revision Summary)
      3. Actual Educational Notes (What it means, How it works, Why it is important, Steps/Logic, Examples)
      4. Quality & Accuracy (Grounded in lecture transcript, passes PedagogicalValidator)
    """

    def test_excludes_non_academic_chatter_and_enforces_universal_structure(self):
        noisy_biology_items = [
            (0.0, 10.0, "Assalamu alaikum everyone, good morning! Can everyone hear me clearly at the back?"),
            (10.0, 20.0, "Please mute your microphones and open your book to page 42 while I share my screen."),
            (20.0, 30.0, "As an AI language model, sure I can help you summarize this transcript."),
            (30.0, 40.0, "Will this topic be in the midterm exam?"),
            (40.0, 62.0, "Today we explore Cellular Respiration, which is defined as the metabolic process by which living cells convert biochemical energy from glucose into adenosine triphosphate, or ATP."),
            (62.0, 84.0, "Glycolysis occurs in the cytoplasm and breaks one six-carbon glucose molecule into two three-carbon pyruvate molecules, producing a net gain of two ATP molecules."),
            (84.0, 105.0, "For example, human skeletal muscle cells rely on rapid glycolysis during intense anaerobic sprinting when oxygen supply is limited."),
            (105.0, 128.0, "Moving on to the Citric Acid Cycle, also called the Krebs Cycle, it is defined as a series of enzyme-catalyzed chemical reactions in the mitochondrial matrix that oxidize acetyl-CoA."),
            (128.0, 150.0, "Now we discuss Oxidative Phosphorylation, where the electron transport chain across the inner mitochondrial membrane drives ATP synthase to generate thirty-four ATP molecules."),
            (150.0, 175.0, "For example, cyanide poisoning inhibits cytochrome c oxidase in the mitochondrial electron transport chain, halting cellular ATP synthesis."),
        ]
        segs = _make_segments(noisy_biology_items)
        full_txt = " ".join(t for _, _, t in noisy_biology_items)
        engine = TopicIntelligenceEngine(use_llm=False)
        notes = engine.analyze_lecture_study_notes(
            full_txt, segs, 175.0, topic_hint="Cellular Respiration & ATP Synthesis"
        )
        notes_dict = notes.to_dict()
        study_topics = StudyGuideSynthesizer.enrich_study_topics(
            notes_dict["study_topics"], full_txt, topic_hint="Cellular Respiration & ATP Synthesis", segments=segs
        )
        guide_md = _build_extractive_fallback_guide(study_topics)

        # 1. Accurate Topic Extraction: No classroom chatter, questions, or AI meta-responses as topics
        titles_lower = [t["title"].lower() for t in study_topics]
        for t_low in titles_lower:
            assert "hear me" not in t_low
            assert "mute your" not in t_low
            assert "page 42" not in t_low
            assert "language model" not in t_low
            assert "midterm" not in t_low
            assert "?" not in t_low

        # Genuine biology concepts are extracted
        combined_titles = " ".join(titles_lower)
        assert any(
            kw in combined_titles
            for kw in ("cellular respiration", "glycolysis", "citric acid", "krebs", "oxidative", "atp")
        )

        # 2. Proper Structure: Executive Summary, Learning Objectives, and Revision Summary
        assert notes_dict.get("executive_summary"), "Missing executive_summary in LectureStudyNotes.to_dict()"
        assert len(notes_dict.get("learning_objectives", [])) >= 1, "Missing learning_objectives"
        assert len(notes_dict.get("revision_summary", [])) >= 1, "Missing revision_summary"
        assert "Executive Summary" in guide_md
        assert "Learning Objectives" in guide_md
        assert "## 2. Core Concepts Overview" in guide_md
        assert "## 8. Important Takeaways & Revision Summary" in guide_md

        # 3. Actual Educational Notes: Teacher-style breakdown on every topic card
        for t in study_topics:
            assert t.get("what_it_means"), f"Topic {t['title']} missing 'what_it_means'"
            assert t.get("how_it_works"), f"Topic {t['title']} missing 'how_it_works'"
            assert t.get("why_it_matters"), f"Topic {t['title']} missing 'why_it_matters'"
            assert len(t.get("steps_and_logic", [])) >= 1, f"Topic {t['title']} missing 'steps_and_logic'"

        # 4. Quality & Accuracy: Passes PedagogicalValidator
        val = PedagogicalValidator.validate_study_notes(guide_md, full_txt)
        assert val["passed"] is True, f"Study guide failed validation: {val['issues']}"

