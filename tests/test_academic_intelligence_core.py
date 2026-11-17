"""
tests/test_academic_intelligence_core.py — Phase 2/3 unit tests: C-value keyphrases, deterministic
importance, evidence alignment, validator bloom_policy, MCQ structure checks, deterministic MCQ engine.

A fake embedder is injected so the suite never loads the real embedding model (fast, deterministic).
"""

from __future__ import annotations

import os
import random
import re
import zlib
import unittest
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("TESTING", "1")

from app.services.academic_preprocess import (
    AcademicPreprocessor, extract_keyphrases, score_importance, importance_label,
)
from app.services.study_guide_generator import AcademicMCQEngine, align_sentence
from app.services.pedagogical_validator import PedagogicalValidator, MCQStructureValidator


# ----------------------------- fake embedders --------------------------------
def bow_embed(texts):
    """Bag-of-words hashing embedder: similar wording -> high cosine."""
    out = np.zeros((len(texts), 128), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z]{3,}", t.lower()):
            out[i, zlib.crc32(w.encode()) % 128] += 1.0
        n = np.linalg.norm(out[i])
        out[i] = out[i] / n if n else out[i]
    return out


def uniform_embed(texts):
    """Every distinct text sits at cosine ~0.5 from every other (inside the distractor band)."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    base = np.ones(64, dtype=np.float32) / 8.0
    for i, t in enumerate(texts):
        r = np.random.RandomState(zlib.crc32(t.encode()) & 0xFFFFFFFF).normal(size=64).astype(np.float32)
        r /= np.linalg.norm(r)
        v = 0.7 * base + 0.7 * r
        out[i] = v / np.linalg.norm(v)
    return out


LECTURE = [
    "Round robin scheduling gives every process in the ready queue a fixed time quantum before switching to the next one.",
    "The time quantum is the maximum amount of CPU time a process may use in one turn.",
    "A context switch saves the state of the running process and loads the state of the next process.",
    "Every context switch adds overhead because the CPU does no useful work while the state is being swapped.",
    "If the time quantum is too large, round robin scheduling behaves almost like first come first served scheduling.",
    "If the time quantum is too small, the context switch overhead dominates and throughput drops.",
    "For example, imagine four processes waiting in the ready queue with a time quantum of four milliseconds.",
    "Turnaround time is the total time from the arrival of a process until its completion.",
    "Waiting time is the turnaround time minus the burst time of the process.",
    "Burst time is the amount of CPU time a process needs to finish its execution.",
    "Priority scheduling always selects the process with the highest priority from the ready queue.",
    "Starvation happens when low priority processes wait forever because higher priority processes keep arriving.",
    "Aging solves starvation by slowly increasing the priority of processes that have waited a long time.",
    "For instance, a process that waits ten minutes might gain one priority level through aging.",
    "The ready queue holds all processes that are loaded in memory and waiting for the CPU.",
    "Round robin scheduling is preemptive because the scheduler interrupts a process when its time quantum expires.",
    "Priority scheduling can also be preemptive when a newly arrived process has a higher priority than the running process.",
    "Average waiting time is calculated by adding the waiting time of every process and dividing by the number of processes.",
    "A shorter burst time does not always mean a shorter turnaround time because of the waiting time in the queue.",
    "Context switch time is not included in the burst time of a process.",
    "Round robin scheduling gives good response time but a higher average turnaround time than shortest job first.",
    "Starvation is not a problem in round robin scheduling because every process eventually receives a time quantum.",
    "Suppose a process has a burst time of ten milliseconds and the time quantum is four milliseconds.",
    "That process needs three turns in the ready queue before its burst time is fully completed.",
    "Turnaround time and waiting time are the two metrics used to compare scheduling algorithms.",
    "The scheduler must balance fairness, response time and throughput when choosing a time quantum.",
    "Because aging raises priority over time, priority scheduling with aging prevents starvation completely.",
    "Therefore the choice of scheduling algorithm depends on the goals of the operating system.",
    "First come first served scheduling runs processes strictly in the order of their arrival time.",
    "The convoy effect occurs in first come first served scheduling when one long process blocks many short processes.",
    "Shortest job first scheduling picks the process with the smallest burst time from the ready queue.",
    "Shortest job first scheduling gives the minimum average waiting time among all scheduling algorithms.",
    "However shortest job first scheduling can cause starvation for processes with a long burst time.",
    "A Gantt chart is a timeline that shows which process runs on the CPU at every moment.",
    "To draw a Gantt chart, start at time zero and place each process according to the scheduling algorithm.",
    "Arrival time is the moment when a process enters the ready queue for the first time.",
    "The dispatcher gives control of the CPU to the process selected by the scheduler.",
    "Dispatch latency is the time the dispatcher needs to stop one process and start another.",
    "Throughput is the number of processes completed per unit of time.",
    "Response time is the time between submitting a request and receiving the first response from the process.",
    "CPU utilization measures how busy the CPU is kept, and a good scheduler keeps it close to full.",
    "Preemptive scheduling can improve response time but it increases the number of context switches.",
    "Non preemptive scheduling lets a process keep the CPU until it finishes or blocks for input and output.",
    "The convoy effect lowers CPU utilization and throughput because short processes wait behind a long burst time.",
]


def make_segments(sentences, per_seg=2, seg_dur=10.0):
    segs = []
    for i in range(0, len(sentences), per_seg):
        k = i // per_seg
        segs.append(SimpleNamespace(
            id=k + 1, segment_index=k, start_time=k * seg_dur, end_time=(k + 1) * seg_dur,
            text=" ".join(sentences[i:i + per_seg]),
        ))
    return segs


def build():
    pre = AcademicPreprocessor.process_segments(make_segments(LECTURE))
    kps = extract_keyphrases(pre.spans, top_k=30)
    return pre, kps


# ------------------------------- keyphrases ----------------------------------
class TestCValue(unittest.TestCase):
    def test_nested_term_does_not_outrank_compound(self):
        segs = [SimpleNamespace(id=i, start_time=i, end_time=i + 1,
                text="Gradient descent updates weights. Gradient descent needs a learning rate.") for i in range(3)]
        pre = AcademicPreprocessor.process_segments(segs)
        kps = dict((k, c) for k, c, _ in extract_keyphrases(pre.spans, min_freq=2))
        self.assertIn("gradient descent", kps)
        self.assertGreater(kps["gradient descent"], kps.get("descent", 0.0))

    def test_candidates_never_cross_segment_boundary(self):
        segs = [SimpleNamespace(id=1, start_time=0, end_time=1, text="alpha beta"),
                SimpleNamespace(id=2, start_time=1, end_time=2, text="gamma delta"),
                SimpleNamespace(id=3, start_time=2, end_time=3, text="alpha beta"),
                SimpleNamespace(id=4, start_time=3, end_time=4, text="gamma delta")]
        pre = AcademicPreprocessor.process_segments(segs)
        phrases = {k for k, _, _ in extract_keyphrases(pre.spans, min_freq=2)}
        self.assertNotIn("beta gamma", phrases)
        self.assertIn("alpha beta", phrases)

    def test_lecture_yields_expected_terms(self):
        _, kps = build()
        phrases = {k for k, _, _ in kps}
        self.assertTrue({"time quantum", "ready queue", "burst time"} <= phrases, phrases)

    def test_deterministic(self):
        self.assertEqual(build()[1], build()[1])


# ------------------------------- importance ----------------------------------
class TestImportance(unittest.TestCase):
    def test_score_in_range_and_label_derived(self):
        pre, kps = build()
        res = score_importance(pre.spans[:4], pre.spans, kps)
        self.assertIn(res["score"], {1, 2, 3, 4, 5})
        self.assertEqual({"time_share", "repetition", "term_density", "structure"}, set(res["components"]))
        self.assertEqual(importance_label(1), "low")
        self.assertEqual(importance_label(3), "medium")
        self.assertEqual(importance_label(5), "high")

    def test_longer_richer_topic_scores_at_least_as_high(self):
        pre, kps = build()
        big = score_importance(pre.spans[:10], pre.spans, kps)["total"]
        small = score_importance(pre.spans[:1], pre.spans, kps)["total"]
        self.assertGreaterEqual(big, small)

    def test_deterministic(self):
        pre, kps = build()
        a = score_importance(pre.spans[:5], pre.spans, kps)
        b = score_importance(pre.spans[:5], pre.spans, kps)
        self.assertEqual(a, b)


# -------------------------------- alignment ----------------------------------
class TestAlignment(unittest.TestCase):
    def test_supported_sentence_aligns_with_time_and_segment(self):
        pre, _ = build()
        res = align_sentence("Starvation happens when low priority processes wait forever.", pre.spans, embed_fn=bow_embed,
                             cos_min=0.3, overlap_min=0.3)
        self.assertIsNotNone(res)
        self.assertTrue(res["segment_ids"])
        self.assertGreaterEqual(res["end_time"], res["start_time"])

    def test_unsupported_sentence_is_rejected(self):
        pre, _ = build()
        res = align_sentence("Mitochondria produce energy inside every living cell.", pre.spans, embed_fn=bow_embed,
                             cos_min=0.3, overlap_min=0.3)
        self.assertIsNone(res)

    def test_window_restricts_search(self):
        pre, _ = build()
        res = align_sentence("Starvation happens when low priority processes wait forever.", pre.spans, embed_fn=bow_embed,
                             cos_min=0.3, overlap_min=0.3, window=(0.0, 5.0))
        self.assertIsNone(res)


# ------------------------------ validator policy -----------------------------
class TestBloomPolicy(unittest.TestCase):
    def _q(self, stem):
        return {"question": stem, "options": ["a1", "b2", "c3", "d4"], "correct_answer": "a1"}

    def test_notes_policy_rejects_recall(self):
        r = PedagogicalValidator.validate_quizzes([self._q("What did the speaker say about queues?")], "queues are fifo")
        self.assertEqual(r["valid_count"], 0)

    def test_quiz_policy_accepts_up_to_two_recall(self):
        qs = [self._q("What did the speaker say about queues?"),
              self._q("What was mentioned about stacks?"),
              self._q("What was said about heaps?")]
        r = PedagogicalValidator.validate_quizzes(qs, "queues stacks heaps", bloom_policy="quiz")
        self.assertEqual(r["valid_count"], 2)
        self.assertEqual(len(r["rejected"]), 1)

    def test_quiz_policy_rejects_verbatim_stem(self):
        transcript = "Round robin scheduling gives every process a fixed time quantum before switching."
        q = self._q("Round robin scheduling gives every process a fixed time quantum before switching")
        r = PedagogicalValidator.validate_quizzes([q], transcript, bloom_policy="quiz")
        self.assertEqual(r["valid_count"], 0)


class TestMCQStructure(unittest.TestCase):
    def _good(self, i=0):
        return {"question": f"Stem {i}?", "options": [f"x{i}a", f"x{i}b", f"x{i}c", f"x{i}d"], "correct_answer": f"x{i}b",
                "explanation": "because", "evidence_timestamps": [{"segment_id": 1, "start": 0, "end": 1}]}

    def test_good_question_passes(self):
        self.assertEqual(MCQStructureValidator.validate_question(self._good()), [])

    def test_catch_all_banned(self):
        q = self._good(); q["options"][2] = "All of the above"
        self.assertIn("catch_all_option", MCQStructureValidator.validate_question(q))

    def test_duplicate_options_flagged(self):
        q = self._good(); q["options"][0] = q["options"][1]
        self.assertIn("duplicate_options", MCQStructureValidator.validate_question(q))

    def test_missing_evidence_flagged(self):
        q = self._good(); q["evidence_timestamps"] = []
        self.assertIn("missing_evidence", MCQStructureValidator.validate_question(q))

    def test_set_level_count(self):
        res = MCQStructureValidator.validate_set([self._good(i) for i in range(10)], expected_count=10)
        self.assertTrue(res["count_ok"])


# ------------------------------- MCQ engine ----------------------------------
class TestMCQEngine(unittest.TestCase):
    """Runs with the DEFAULT MIN_WORDS_FOR_10 (600): the fixture lecture must genuinely exceed it."""

    def test_fixture_is_long_enough_for_default_policy(self):
        import app.config as cfg
        pre, _ = build()
        self.assertEqual(cfg.MIN_WORDS_FOR_10, 600)
        self.assertGreaterEqual(sum(len(s.tokens) for s in pre.spans), cfg.MIN_WORDS_FOR_10)

    def _gen(self, seed="h1"):
        pre, kps = build()
        return AcademicMCQEngine.generate(pre.spans, kps, rng=random.Random(seed), embed_fn=uniform_embed, transcript_hash=seed)

    def test_exactly_ten_when_content_sufficient(self):
        res = self._gen()
        self.assertEqual(res["mcq_count"], 10, res)
        self.assertEqual(res["mcq_status"], "complete")

    def test_all_structural_invariants_hold(self):
        res = self._gen()
        v = MCQStructureValidator.validate_set(res["mcqs"], expected_count=10)
        self.assertTrue(v["count_ok"])
        self.assertEqual(v["problems"], [], v["problems"])
        self.assertEqual(v["duplicate_stems"], 0)

    def test_answer_is_in_source_and_not_leaked(self):
        pre, kps = build()
        res = AcademicMCQEngine.generate(pre.spans, kps, rng=random.Random("x"), embed_fn=uniform_embed)
        for q in res["mcqs"]:
            self.assertNotIn(q["correct_answer"].lower(), re.findall(r'"([^"]+)"', q["question"])[0].lower())
            self.assertIn(q["correct_answer"].lower(), q["explanation"].lower())

    def test_answer_positions_balanced(self):
        counts = MCQStructureValidator.validate_set(self._gen()["mcqs"])["answer_positions"]
        self.assertLessEqual(max(counts) - min(counts), 2)

    def test_reproducible_with_same_seed(self):
        a = [q["question"] for q in self._gen("same")["mcqs"]]
        b = [q["question"] for q in self._gen("same")["mcqs"]]
        self.assertEqual(a, b)
        self.assertEqual([q["options"] for q in self._gen("same")["mcqs"]], [q["options"] for q in self._gen("same")["mcqs"]])

    def test_no_padding_on_short_lecture(self):
        segs = make_segments(LECTURE[:6])
        pre = AcademicPreprocessor.process_segments(segs)
        kps = extract_keyphrases(pre.spans, min_freq=2)
        res = AcademicMCQEngine.generate(pre.spans, kps, rng=random.Random("s"), embed_fn=uniform_embed)
        self.assertEqual(res["mcq_status"], "insufficient_content")
        self.assertLess(res["mcq_count"], 10)
        self.assertIn(res["reason"], {"too_short", "too_few_concepts", "validation_attrition"})

    def test_short_lecture_reports_too_short_by_default_policy(self):
        pre = AcademicPreprocessor.process_segments(make_segments(LECTURE[:6]))
        kps = extract_keyphrases(pre.spans, min_freq=2)
        res = AcademicMCQEngine.generate(pre.spans, kps, rng=random.Random("d"), embed_fn=uniform_embed)
        self.assertEqual(res["mcq_status"], "insufficient_content")
        self.assertEqual(res["reason"], "too_short")

    def test_negation_flip_rejected(self):
        src = "Context switch time is not included in the burst time of a process."
        q = {"question": 'Which term completes this? "Context switch time is included in the _____ of a process."',
             "correct_answer": "burst time", "explanation": ""}
        self.assertIn("negation_polarity_mismatch", AcademicMCQEngine.verify_question(q, src))

    def test_answer_leak_rejected(self):
        src = "Burst time is the amount of CPU time a process needs."
        q = {"question": 'Which term? "Burst time is the amount of CPU time. burst time again _____"',
             "correct_answer": "burst time", "explanation": ""}
        self.assertIn("answer_leaked_in_stem", AcademicMCQEngine.verify_question(q, src))

    def test_no_generic_distractors(self):
        for q in self._gen()["mcqs"]:
            self.assertFalse(any(re.match(r"^option\s*[a-d]$", o, re.I) for o in q["options"]))
            self.assertEqual(len(set(q["options"])), 4)


if __name__ == "__main__":
    unittest.main()
