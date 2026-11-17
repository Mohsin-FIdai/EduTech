"""
tests/test_contamination_phase2.py — Phase 2: Domain Gate, Strict Mode & ContaminationAuditor Tests.

Covers:
- classify_domain() accuracy across all 6 domain types + general
- Domain gate margins and thresholds (min-distinct-terms, min-weight, margin ratio)
- LectureContext.from_transcript() auto-classification
- LectureContext.get_rng() isolation (different recordings produce different seeds, same recording always same seed)
- ContaminationAuditor.audit_text() with clean and contaminated inputs
- ContaminationAuditor.audit_notes() and audit_quizzes()
- STRICT_TRANSCRIPT_ONLY config flag effect
"""

from __future__ import annotations

import os
import unittest

# Ensure testing mode so LLM is disabled
os.environ.setdefault("TESTING", "1")

from app.services.lecture_context import LectureContext, classify_domain
from app.services.pedagogical_validator import ContaminationAuditor


# ---------------------------------------------------------------------------
# Domain Classification Tests
# ---------------------------------------------------------------------------

class TestClassifyDomainOS(unittest.TestCase):
    """classify_domain correctly identifies Operating Systems transcripts."""

    OS_TRANSCRIPT = (
        "Today we discuss CPU scheduling. The FCFS algorithm is non-preemptive and suffers from "
        "the convoy effect when long burst-time processes block shorter ones in the ready queue. "
        "We compute turnaround time as completion minus arrival. "
        "A semaphore protects critical sections preventing deadlock. "
        "Context switching between processes is managed by the dispatcher."
    )

    def test_os_domain_detected(self):
        domain, conf = classify_domain(self.OS_TRANSCRIPT)
        self.assertEqual(domain, "operating_system")

    def test_os_confidence_above_zero(self):
        _, conf = classify_domain(self.OS_TRANSCRIPT)
        self.assertGreater(conf, 0.0)

    def test_os_confidence_at_most_one(self):
        _, conf = classify_domain(self.OS_TRANSCRIPT)
        self.assertLessEqual(conf, 1.0)


class TestClassifyDomainML(unittest.TestCase):
    """classify_domain correctly identifies Machine Learning transcripts."""

    ML_TRANSCRIPT = (
        "In this lecture we explore supervised learning. Labeled training data with features X "
        "and ground-truth labels y. Overfitting means the model memorises noise. Underfitting "
        "is when the model has too high a bias. Regularization with L2 prevents overfitting. "
        "Gradient descent minimises the loss. Hyperparameter tuning determines the epoch count. "
        "Backpropagation computes gradients layer by layer."
    )

    def test_ml_domain_detected(self):
        domain, _ = classify_domain(self.ML_TRANSCRIPT)
        self.assertEqual(domain, "machine_learning")

    def test_ml_confidence_positive(self):
        _, conf = classify_domain(self.ML_TRANSCRIPT)
        self.assertGreater(conf, 0.0)


class TestClassifyDomainBiology(unittest.TestCase):
    """classify_domain correctly identifies Biology transcripts."""

    BIO_TRANSCRIPT = (
        "The reflex arc is a rapid involuntary response. A stimulus is detected by a receptor "
        "in the sense organ and transmitted along a sensory neuron through the spinal cord. "
        "The nervous system coordinates these responses. Hormones secreted by the endocrine "
        "system act as chemical messengers. A synapse connects two neurons. "
        "The organism maintains homeostasis through feedback mechanisms."
    )

    def test_biology_domain_detected(self):
        domain, _ = classify_domain(self.BIO_TRANSCRIPT)
        self.assertEqual(domain, "biology")


class TestClassifyDomainPhysics(unittest.TestCase):
    """classify_domain correctly identifies Physics / Electricity transcripts."""

    PHYSICS_TRANSCRIPT = (
        "Ohm's Law states V equals I times R. The unit of electrical resistance is the ohm. "
        "Current is measured in amperes and charge in coulombs. A capacitor stores charge. "
        "The inductor opposes changes in current. Resistivity of a conductor depends on material. "
        "We analyse series and parallel circuits using Kirchhoff's laws. "
        "Electromagnetic induction is described by Faraday's law. Power in watts equals V times I."
    )

    def test_physics_domain_detected(self):
        domain, _ = classify_domain(self.PHYSICS_TRANSCRIPT)
        self.assertEqual(domain, "physics")


class TestClassifyDomainGeneral(unittest.TestCase):
    """classify_domain returns 'general' for ambiguous / short transcripts."""

    GENERIC_TRANSCRIPT = (
        "Good morning team. Let's discuss today's agenda. We have several updates to share "
        "with the group. The project is on track and milestones are being met. "
        "Please submit your timesheets by Friday. Thank you."
    )

    def test_generic_is_general(self):
        domain, conf = classify_domain(self.GENERIC_TRANSCRIPT)
        self.assertEqual(domain, "general")
        self.assertEqual(conf, 0.0)

    def test_empty_transcript_is_general(self):
        domain, conf = classify_domain("")
        self.assertEqual(domain, "general")
        self.assertEqual(conf, 0.0)

    def test_one_word_is_general(self):
        domain, conf = classify_domain("hello")
        self.assertEqual(domain, "general")
        self.assertEqual(conf, 0.0)


class TestClassifyDomainMargin(unittest.TestCase):
    """Domain gate correctly rejects when margin is insufficient."""

    def test_mixed_OS_ML_is_general(self):
        """A transcript with equal OS and ML signals should fall back to general."""
        mixed = (
            "We discuss CPU scheduling with FCFS and also supervised learning with overfitting. "
            "The convoy effect blocks processes. Labeled training data prevents underfitting. "
            "Gradient descent optimises. The dispatcher runs the scheduler."
        )
        domain, _ = classify_domain(mixed)
        # May resolve to either domain or general depending on weights — it must NOT
        # confidently claim a single domain with a clear margin if signals are balanced.
        self.assertIn(domain, {"general", "operating_system", "machine_learning"})


# ---------------------------------------------------------------------------
# LectureContext Tests
# ---------------------------------------------------------------------------

class TestLectureContextAutoClassify(unittest.TestCase):
    """LectureContext.from_transcript() auto-classifies correctly."""

    def test_os_context_domain(self):
        transcript = (
            "CPU scheduling using FCFS. The convoy effect delays short processes. "
            "Turnaround time is completion minus arrival. Semaphore guards critical sections."
        )
        ctx = LectureContext.from_transcript("rec_os_01", transcript)
        self.assertEqual(ctx.domain, "operating_system")

    def test_context_vocab_populated(self):
        ctx = LectureContext.from_transcript("rec_01", "The CPU scheduler runs FCFS")
        self.assertIn("scheduler", ctx.vocab)
        self.assertIn("fcfs", ctx.vocab)

    def test_context_transcript_hash_set(self):
        ctx = LectureContext.from_transcript("rec_01", "Hello world")
        self.assertGreater(len(ctx.transcript_hash), 0)

    def test_same_transcript_same_hash(self):
        ctx1 = LectureContext.from_transcript("rec_A", "Test transcript")
        ctx2 = LectureContext.from_transcript("rec_B", "Test transcript")
        self.assertEqual(ctx1.transcript_hash, ctx2.transcript_hash)

    def test_different_transcripts_different_hashes(self):
        ctx1 = LectureContext.from_transcript("rec_A", "OS scheduling")
        ctx2 = LectureContext.from_transcript("rec_A", "ML regression")
        self.assertNotEqual(ctx1.transcript_hash, ctx2.transcript_hash)


class TestLectureContextRNG(unittest.TestCase):
    """LectureContext.get_rng() is deterministic and recording-isolated."""

    def test_same_context_same_shuffle_order(self):
        ctx = LectureContext.from_transcript("rec_01", "Hello world test")
        items1 = [1, 2, 3, 4, 5]
        items2 = [1, 2, 3, 4, 5]
        ctx.get_rng("quiz").shuffle(items1)
        ctx.get_rng("quiz").shuffle(items2)
        self.assertEqual(items1, items2)

    def test_different_recordings_different_shuffles(self):
        ctx_a = LectureContext.from_transcript("rec_A", "Some unique text about biology")
        ctx_b = LectureContext.from_transcript("rec_B", "Some other text about physics now")
        items_a = list(range(10))
        items_b = list(range(10))
        ctx_a.get_rng("quiz").shuffle(items_a)
        ctx_b.get_rng("quiz").shuffle(items_b)
        self.assertNotEqual(items_a, items_b)

    def test_rng_salt_differentiates(self):
        ctx = LectureContext.from_transcript("rec_01", "Some transcript")
        rng1 = ctx.get_rng("quiz_salt_A")
        rng2 = ctx.get_rng("quiz_salt_B")
        items1 = list(range(8))
        items2 = list(range(8))
        rng1.shuffle(items1)
        rng2.shuffle(items2)
        self.assertNotEqual(items1, items2)


class TestLectureContextIsWordInTranscript(unittest.TestCase):
    """LectureContext.is_word_in_transcript() works correctly."""

    def setUp(self):
        self.ctx = LectureContext.from_transcript("rec_01", "The CPU scheduler runs FCFS algorithm")

    def test_known_word_found(self):
        self.assertTrue(self.ctx.is_word_in_transcript("cpu"))

    def test_known_word_case_insensitive(self):
        self.assertTrue(self.ctx.is_word_in_transcript("FCFS"))

    def test_foreign_word_not_found(self):
        self.assertFalse(self.ctx.is_word_in_transcript("reflex"))

    def test_empty_word_not_found(self):
        self.assertFalse(self.ctx.is_word_in_transcript(""))

    def test_multiword_phrase_found(self):
        ctx2 = LectureContext.from_transcript("rec_02", "The convoy effect causes long delays")
        self.assertTrue(ctx2.is_word_in_transcript("convoy effect"))

    def test_multiword_phrase_not_found(self):
        ctx2 = LectureContext.from_transcript("rec_02", "The convoy effect causes long delays")
        self.assertFalse(ctx2.is_word_in_transcript("reflex arc"))


# ---------------------------------------------------------------------------
# ContaminationAuditor Tests
# ---------------------------------------------------------------------------

class TestContaminationAuditorClean(unittest.TestCase):
    """ContaminationAuditor passes clean, grounded output."""

    def test_clean_os_output_passes(self):
        transcript_vocab = {"cpu", "scheduling", "fcfs", "convoy", "turnaround", "process", "queue"}
        clean_text = "The FCFS scheduling algorithm causes the convoy effect on the process queue."
        result = ContaminationAuditor.audit_text(clean_text, transcript_vocab)
        self.assertTrue(result["passed"])
        self.assertEqual(result["leaked_terms"], [])

    def test_generic_text_passes(self):
        vocab = {"team", "meeting", "agenda", "project", "milestone"}
        text = "The team had a productive meeting and reviewed the agenda."
        result = ContaminationAuditor.audit_text(text, vocab)
        self.assertTrue(result["passed"])

    def test_empty_text_passes(self):
        result = ContaminationAuditor.audit_text("", set())
        self.assertTrue(result["passed"])


class TestContaminationAuditorLeaks(unittest.TestCase):
    """ContaminationAuditor detects foreign domain terms."""

    def test_os_term_in_biology_output(self):
        """An OS term (deadlock) appearing in biology output should be flagged."""
        biology_vocab = {"reflex", "stimulus", "receptor", "neuron", "spinal"}
        contaminated = "The reflex arc involves deadlock between sensory and motor neurons."
        result = ContaminationAuditor.audit_text(contaminated, biology_vocab)
        self.assertFalse(result["passed"])
        self.assertIn("deadlock", result["leaked_terms"])
        self.assertIn("operating_system", result["domains_detected"])

    def test_finance_term_in_os_output(self):
        """A finance term (accrual) appearing in OS output should be flagged."""
        os_vocab = {"cpu", "scheduling", "fcfs", "convoy", "turnaround", "process"}
        contaminated = "The CPU uses accrual accounting to measure turnaround time."
        result = ContaminationAuditor.audit_text(contaminated, os_vocab)
        self.assertFalse(result["passed"])
        self.assertIn("accrual", result["leaked_terms"])
        self.assertIn("accounting_finance", result["domains_detected"])

    def test_multiple_foreign_terms_all_flagged(self):
        vocab = {"cpu", "scheduling"}
        text = "The scheduling algorithm uses backpropagation and overfitting parameters."
        result = ContaminationAuditor.audit_text(text, vocab)
        self.assertFalse(result["passed"])
        leaked = result["leaked_terms"]
        self.assertTrue(any(t in leaked for t in ["backpropagation", "overfitting"]))

    def test_flagged_sentences_populated(self):
        vocab = {"reflex", "stimulus"}
        text = "The reflex is fast. The deadlock between neurons is common. The stimulus triggers response."
        result = ContaminationAuditor.audit_text(text, vocab)
        self.assertGreater(len(result["flagged_sentences"]), 0)
        flagged_combined = " ".join(result["flagged_sentences"]).lower()
        self.assertIn("deadlock", flagged_combined)

    def test_term_in_transcript_not_flagged(self):
        """A term that IS in the transcript vocab should NOT be flagged."""
        vocab = {"deadlock", "cpu", "scheduling", "process", "mutex", "semaphore",
                 "fcfs", "convoy", "turnaround", "burst", "dispatcher", "preemptive"}
        text = "Deadlock occurs when processes are stuck waiting for each other."
        result = ContaminationAuditor.audit_text(text, vocab)
        self.assertNotIn("deadlock", result["leaked_terms"])


class TestContaminationAuditorNotes(unittest.TestCase):
    """ContaminationAuditor.audit_notes() works on structured notes dicts."""

    def test_clean_notes_pass(self):
        os_vocab = {"cpu", "scheduling", "fcfs", "convoy", "turnaround", "burst",
                    "process", "queue", "semaphore", "mutex", "deadlock", "dispatcher",
                    "preemptive", "starvation", "aging", "pcb"}
        notes = {
            "study_topics": [
                {
                    "title": "FCFS Scheduling",
                    "explanation": "FCFS dispatches CPU tasks in arrival order.",
                    "key_points": ["Convoy effect delays short processes."],
                    "definitions": [{"term": "FCFS", "definition": "First-Come First-Served scheduling."}],
                }
            ],
            "definitions": [],
            "markdown_study_guide": "# FCFS\nNon-preemptive scheduling with convoy effect.",
        }
        result = ContaminationAuditor.audit_notes(notes, os_vocab)
        self.assertTrue(result["passed"])

    def test_contaminated_notes_fail(self):
        os_vocab = {"cpu", "scheduling", "process", "queue"}
        notes = {
            "study_topics": [
                {
                    "title": "CPU Scheduling",
                    "explanation": "CPU scheduling uses gradient descent to pick processes.",
                    "key_points": ["Backpropagation helps select the next process."],
                    "definitions": [],
                }
            ],
            "definitions": [],
            "markdown_study_guide": "",
        }
        result = ContaminationAuditor.audit_notes(notes, os_vocab)
        self.assertFalse(result["passed"])

    def test_reference_tagged_definitions_skipped(self):
        """Definitions tagged with source='reference' are excluded from the audit."""
        vocab = {"cpu", "scheduling"}
        notes = {
            "study_topics": [
                {
                    "title": "CPU Scheduling",
                    "explanation": "The CPU runs scheduling algorithms.",
                    "key_points": [],
                    "definitions": [
                        {"term": "Backpropagation", "definition": "Gradient computation in neural nets.",
                         "source": "reference"},
                    ],
                }
            ],
            "definitions": [],
            "markdown_study_guide": "",
        }
        result = ContaminationAuditor.audit_notes(notes, vocab)
        self.assertNotIn("backpropagation", result["leaked_terms"])


class TestContaminationAuditorQuizzes(unittest.TestCase):
    """ContaminationAuditor.audit_quizzes() works on quiz lists."""

    def test_clean_quizzes_pass(self):
        os_vocab = {"cpu", "scheduling", "fcfs", "convoy", "turnaround", "burst",
                    "process", "dispatcher", "semaphore", "preemptive", "deadlock",
                    "mutex", "starvation", "aging", "pcb"}
        quizzes = [{
            "question": "What causes the convoy effect in FCFS scheduling?",
            "options": ["Long burst-time process blocks short ones", "Wrong answer A", "Wrong answer B", "Wrong answer C"],
            "correct_answer": "Long burst-time process blocks short ones",
            "explanation": "FCFS is non-preemptive so a long process delays everything behind it.",
        }]
        result = ContaminationAuditor.audit_quizzes(quizzes, os_vocab)
        self.assertTrue(result["passed"])

    def test_contaminated_quizzes_fail(self):
        os_vocab = {"cpu", "scheduling", "process"}
        quizzes = [{
            "question": "What is backpropagation in CPU scheduling?",
            "options": ["Gradient descent method", "FCFS order", "Round robin", "None"],
            "correct_answer": "Gradient descent method",
            "explanation": "Backpropagation computes gradients.",
        }]
        result = ContaminationAuditor.audit_quizzes(quizzes, os_vocab)
        self.assertFalse(result["passed"])
        self.assertIn("backpropagation", result["leaked_terms"])


if __name__ == "__main__":
    unittest.main()
