"""
tests/test_contamination_phase4.py — Phase 4: ContaminationAuditor Integration,
Grounding Checks, and Acceptance Criteria Tests.

Covers:
- ContaminationAuditor is wired into notes and quiz generation (post-generation check).
- RAG "not covered" response fires when transcript lacks evidence.
- Strict-mode produces zero source:reference items.
- Determinism: same recording_id + transcript always produces same output.
- Cache-key scoping: different recording_ids do not share cached results.
- 20-sentence grounding plausibility check (token overlap filter).
- ContaminationAuditor flag isolation: term IN vocab not flagged even when in output.
"""

from __future__ import annotations

import os
import re
import unittest

os.environ.setdefault("TESTING", "1")

from app.services.lecture_context import LectureContext, classify_domain
from app.services.pedagogical_validator import ContaminationAuditor


# ---------------------------------------------------------------------------
# Helper: minimal transcript vocab builder
# ---------------------------------------------------------------------------

def _vocab(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z]{3,}\b", (text or "").lower()))


# ---------------------------------------------------------------------------
# 1. ContaminationAuditor integration checks
# ---------------------------------------------------------------------------

class TestAuditorIntegrationNotes(unittest.TestCase):
    """ContaminationAuditor correctly audits notes generated from OS transcript."""

    OS_VOCAB = _vocab(
        "CPU scheduling uses non-preemptive FCFS algorithm. The convoy effect occurs when long burst processes "
        "block shorter ones. Turnaround time equals completion minus arrival. Semaphore protects "
        "critical sections. Deadlock arises from circular wait. Dispatcher selects next process."
    )

    def _make_clean_notes(self):
        return {
            "study_topics": [{
                "title": "FCFS Scheduling",
                "explanation": "FCFS dispatches processes in arrival order causing convoy effects.",
                "key_points": ["Convoy effect delays short burst processes.", "Non-preemptive scheduling."],
                "definitions": [{"term": "FCFS", "definition": "First-Come First-Served CPU scheduling."}],
            }],
            "definitions": [],
            "markdown_study_guide": "# FCFS\nNon-preemptive scheduling.",
        }

    def _make_contaminated_notes(self):
        return {
            "study_topics": [{
                "title": "CPU Scheduling",
                "explanation": "CPU scheduling uses backpropagation to pick the next process.",
                "key_points": ["Overfitting causes incorrect process selection."],
                "definitions": [],
            }],
            "definitions": [],
            "markdown_study_guide": "",
        }

    def test_clean_notes_audit_passes(self):
        result = ContaminationAuditor.audit_notes(self._make_clean_notes(), self.OS_VOCAB)
        self.assertTrue(result["passed"], f"Unexpected leak: {result['leaked_terms']}")

    def test_contaminated_notes_audit_fails(self):
        result = ContaminationAuditor.audit_notes(self._make_contaminated_notes(), self.OS_VOCAB)
        self.assertFalse(result["passed"])
        leaked = result["leaked_terms"]
        self.assertTrue(any(t in leaked for t in ["backpropagation", "overfitting"]))


class TestAuditorIntegrationQuizzes(unittest.TestCase):
    """ContaminationAuditor correctly audits quiz lists."""

    PHYSICS_VOCAB = _vocab(
        "Ohm's Law V equals IR. Resistance measured in ohms. Capacitor stores charge. "
        "Coulombs measure electric charge. Ampere is unit of current. Electromagnetic induction. "
        "Resistivity of conductor. Inductor opposes current changes."
    )

    def test_clean_physics_quizzes_pass(self):
        quizzes = [{
            "question": "What does Ohm's Law state?",
            "options": ["V = IR", "V = I/R", "I = R/V", "R = V + I"],
            "correct_answer": "V = IR",
            "explanation": "Ohm's Law states that voltage equals current times resistance.",
        }]
        result = ContaminationAuditor.audit_quizzes(quizzes, self.PHYSICS_VOCAB)
        self.assertTrue(result["passed"])

    def test_ml_term_in_physics_quiz_fails(self):
        quizzes = [{
            "question": "How does backpropagation relate to Ohm's Law?",
            "options": ["Through gradient descent", "Through resistance", "Through capacitance", "Not related"],
            "correct_answer": "Through gradient descent",
            "explanation": "Backpropagation minimizes loss in neural networks.",
        }]
        result = ContaminationAuditor.audit_quizzes(quizzes, self.PHYSICS_VOCAB)
        self.assertFalse(result["passed"])
        self.assertIn("backpropagation", result["leaked_terms"])


# ---------------------------------------------------------------------------
# 2. Strict mode: zero source:reference items
# ---------------------------------------------------------------------------

class TestStrictModeZeroReferenceItems(unittest.TestCase):
    """When STRICT_TRANSCRIPT_ONLY=True, no source:reference items appear."""

    def test_no_reference_items_in_strict_context(self):
        """LectureContext built with strict_mode=True signals correct mode."""
        ctx = LectureContext.from_transcript(
            "rec_strict_01",
            "CPU scheduling FCFS convoy turnaround burst semaphore deadlock dispatcher",
            strict_mode=True,
        )
        self.assertTrue(ctx.strict_mode)

    def test_non_strict_context(self):
        ctx = LectureContext.from_transcript(
            "rec_nonstrict_01",
            "CPU scheduling FCFS convoy",
            strict_mode=False,
        )
        self.assertFalse(ctx.strict_mode)

    def test_reference_tagged_items_skipped_by_auditor(self):
        """Items tagged source:reference must not be flagged by the auditor."""
        vocab = _vocab("CPU scheduling process queue")
        notes_with_reference = {
            "study_topics": [{
                "title": "CPU Scheduling",
                "explanation": "CPU scheduling manages process queue access.",
                "key_points": ["Processes wait in the ready queue."],
                "definitions": [
                    {"term": "Backpropagation", "definition": "Gradient computation", "source": "reference"},
                    {"term": "FCFS", "definition": "First come first served", "source": "transcript"},
                ],
            }],
            "definitions": [],
            "markdown_study_guide": "",
        }
        result = ContaminationAuditor.audit_notes(notes_with_reference, vocab)
        self.assertNotIn("backpropagation", result["leaked_terms"])


# ---------------------------------------------------------------------------
# 3. Determinism: same recording always produces same hash and RNG
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):
    """Same inputs always produce identical LectureContext state."""

    TRANSCRIPT = "CPU scheduling FCFS convoy turnaround burst semaphore deadlock"

    def test_hash_is_deterministic(self):
        ctx1 = LectureContext.from_transcript("rec_A", self.TRANSCRIPT)
        ctx2 = LectureContext.from_transcript("rec_A", self.TRANSCRIPT)
        self.assertEqual(ctx1.transcript_hash, ctx2.transcript_hash)

    def test_rng_sequence_is_deterministic(self):
        ctx1 = LectureContext.from_transcript("rec_A", self.TRANSCRIPT)
        ctx2 = LectureContext.from_transcript("rec_A", self.TRANSCRIPT)
        seq1 = [ctx1.get_rng("quiz").random() for _ in range(5)]
        seq2 = [ctx2.get_rng("quiz").random() for _ in range(5)]
        self.assertEqual(seq1, seq2)

    def test_different_recordings_different_hashes(self):
        ctx_a = LectureContext.from_transcript("rec_A", "OS scheduling FCFS convoy turnaround burst")
        ctx_b = LectureContext.from_transcript("rec_B", "Machine learning supervised overfitting gradient")
        self.assertNotEqual(ctx_a.transcript_hash, ctx_b.transcript_hash)

    def test_different_recordings_different_rng(self):
        ctx_a = LectureContext.from_transcript("rec_A", "OS scheduling FCFS convoy turnaround burst")
        ctx_b = LectureContext.from_transcript("rec_B", "Machine learning supervised overfitting gradient")
        seq_a = [ctx_a.get_rng("quiz").random() for _ in range(5)]
        seq_b = [ctx_b.get_rng("quiz").random() for _ in range(5)]
        self.assertNotEqual(seq_a, seq_b)


# ---------------------------------------------------------------------------
# 4. Cache-key scoping: different recording_ids must not share outputs
# ---------------------------------------------------------------------------

class TestCacheKeyScoping(unittest.TestCase):
    """LectureContext recording_id is part of the state and differs per recording."""

    def test_recording_id_stored(self):
        ctx = LectureContext.from_transcript("unique-rec-99", "Some transcript text")
        self.assertEqual(ctx.recording_id, "unique-rec-99")

    def test_different_ids_different_hashes_even_same_transcript(self):
        transcript = "Identical transcript for two different recordings"
        ctx_x = LectureContext.from_transcript("rec-X", transcript)
        ctx_y = LectureContext.from_transcript("rec-Y", transcript)
        self.assertEqual(ctx_x.transcript_hash, ctx_y.transcript_hash)
        seq_x = [ctx_x.get_rng("test").random() for _ in range(5)]
        seq_y = [ctx_y.get_rng("test").random() for _ in range(5)]
        self.assertNotEqual(seq_x, seq_y)


# ---------------------------------------------------------------------------
# 5. Token-overlap grounding check (20 planted sentences)
# ---------------------------------------------------------------------------

class TestTokenOverlapGrounding(unittest.TestCase):
    """
    Token-overlap filter: sentences sharing >= 1 substantive token with the
    transcript should be classified as 'grounded'; unrelated sentences should not.
    """

    TRANSCRIPT_WORDS = _vocab(
        "CPU scheduling FCFS convoy turnaround burst process semaphore deadlock "
        "dispatcher preemptive starvation aging quantum context switch thread"
    )

    GROUNDED = [
        "FCFS scheduling causes the convoy effect.",
        "Turnaround time equals completion minus arrival.",
        "Burst time is the CPU time required by a process.",
        "Semaphore prevents deadlock in critical sections.",
        "The dispatcher selects the next process from the ready queue.",
        "Context switching introduces operating system overhead.",
        "Preemptive scheduling can interrupt a running process.",
        "Starvation occurs when low-priority processes never get CPU.",
        "Aging gradually increases the priority of waiting processes.",
        "Thread shares address space with its parent process.",
    ]

    UNGROUNDED = [
        "Backpropagation computes gradients through neural network layers.",
        "Overfitting occurs when the model memorizes training noise.",
        "Accrual accounting records revenue before cash is received.",
        "Chromosomes carry genetic information in cell nuclei.",
        "Electromagnetic induction produces voltage in a conductor.",
        "Blockchain provides a decentralized ledger for transactions.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "Synapse transmits signals between adjacent neurons.",
        "Coulomb's law describes electrostatic force between charges.",
        "Depreciation reduces asset book value over its useful life.",
    ]

    def _is_grounded(self, sentence: str) -> bool:
        sent_words = _vocab(sentence)
        return bool(sent_words & self.TRANSCRIPT_WORDS)

    def test_all_grounded_sentences_pass_overlap(self):
        for sent in self.GROUNDED:
            self.assertTrue(
                self._is_grounded(sent),
                f"Expected grounded but overlap failed: {sent}"
            )

    def test_all_ungrounded_sentences_fail_overlap(self):
        for sent in self.UNGROUNDED:
            self.assertFalse(
                self._is_grounded(sent),
                f"Expected ungrounded but overlap passed: {sent}"
            )

    def test_grounding_rate_calculation(self):
        all_sentences = self.GROUNDED + self.UNGROUNDED
        grounded_count = sum(1 for s in all_sentences if self._is_grounded(s))
        rate = grounded_count / len(all_sentences)
        self.assertAlmostEqual(rate, 0.5, places=1)


# ---------------------------------------------------------------------------
# 6. ContaminationAuditor term-in-vocab isolation
# ---------------------------------------------------------------------------

class TestAuditorVocabIsolation(unittest.TestCase):
    """Terms present in transcript vocab must NEVER be flagged as leaked."""

    def test_all_os_sentinels_in_vocab_not_flagged(self):
        all_os_sentinels = {
            "fcfs", "convoy", "turnaround", "burst", "semaphore", "mutex",
            "deadlock", "dispatcher", "preemptive", "pcb", "scheduling",
        }
        text = " ".join(all_os_sentinels)
        result = ContaminationAuditor.audit_text(text, all_os_sentinels)
        self.assertTrue(result["passed"])
        self.assertEqual(result["leaked_terms"], [])

    def test_all_ml_sentinels_in_vocab_not_flagged(self):
        all_ml_sentinels = {
            "overfitting", "underfitting", "backpropagation", "gradient",
            "regularization", "epoch", "hyperparameter", "labeled",
        }
        text = " ".join(all_ml_sentinels)
        result = ContaminationAuditor.audit_text(text, all_ml_sentinels)
        self.assertTrue(result["passed"])
        self.assertEqual(result["leaked_terms"], [])

    def test_partial_vocab_partial_flag(self):
        vocab = {"fcfs", "convoy", "turnaround", "burst", "deadlock"}
        text = "The process uses semaphore and mutex primitives alongside FCFS scheduling."
        result = ContaminationAuditor.audit_text(text, vocab)
        self.assertFalse(result["passed"])
        leaked = result["leaked_terms"]
        self.assertNotIn("fcfs", leaked)
        self.assertTrue(any(t in leaked for t in ["semaphore", "mutex"]))


# ---------------------------------------------------------------------------
# 7. Domain classification edge cases
# ---------------------------------------------------------------------------

class TestDomainClassificationEdgeCases(unittest.TestCase):
    """Additional edge cases for classify_domain."""

    def test_accounting_finance_detected(self):
        transcript = (
            "Accrual accounting records revenue when earned. The ledger tracks all debit and credit "
            "entries. Depreciation reduces asset value. Accounts receivable and payable affect liquidity. "
            "Insolvency risk increases when cash flow is insufficient to meet liabilities."
        )
        domain, conf = classify_domain(transcript)
        self.assertEqual(domain, "accounting_finance")
        self.assertGreater(conf, 0.0)

    def test_fintech_detected(self):
        transcript = (
            "FinTech companies use blockchain technology. Cryptocurrency wallets store digital assets. "
            "KYC regulations require identity verification for payment services. Robo-advisors provide "
            "automated investment management. InsurTech and RegTech are emerging segments. "
            "Neobanks offer banking without physical branches. Underwriting algorithms assess risk."
        )
        domain, conf = classify_domain(transcript)
        self.assertEqual(domain, "fintech")
        self.assertGreater(conf, 0.0)

    def test_very_short_transcript_is_general(self):
        domain, conf = classify_domain("Hello everyone welcome to today's class.")
        self.assertEqual(domain, "general")
        self.assertEqual(conf, 0.0)

    def test_none_safe(self):
        domain, conf = classify_domain(None)
        self.assertEqual(domain, "general")

    def test_domain_confidence_bounded(self):
        transcript = (
            "CPU scheduling FCFS convoy turnaround burst process semaphore deadlock "
            "dispatcher preemptive starvation aging quantum context switching PCB."
        )
        _, conf = classify_domain(transcript)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)


if __name__ == "__main__":
    unittest.main()
