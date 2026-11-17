"""
app/services/lecture_context.py — Per-Request Isolation Context for Lecture Processing.

Provides strict boundaries and deterministic scoping across:
1. Recording ID & Transcript Hash
2. Domain Classification (Gate)
3. Recording Vocabulary
4. Isolated local RNG (seeded from recording + transcript hash)
5. Strict mode configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import random
import re
from typing import Optional, Set


@dataclass
class LectureContext:
    """Per-request context ensuring complete isolation across recordings."""
    recording_id: str
    transcript_hash: str = ""
    domain: str = "general"
    confidence: float = 0.0
    vocab: Set[str] = field(default_factory=set)
    strict_mode: bool = True

    @classmethod
    def from_transcript(
        cls,
        recording_id: str,
        transcript_text: str,
        domain: Optional[str] = None,
        confidence: Optional[float] = None,
        strict_mode: Optional[bool] = None,
    ) -> "LectureContext":
        """Factory creating a fully initialized LectureContext from transcript text.
        If domain is None, auto-classifies using classify_domain().
        """
        from app.config import STRICT_TRANSCRIPT_ONLY

        t_clean = (transcript_text or "").strip()
        t_hash = hashlib.sha256(t_clean.encode("utf-8")).hexdigest() if t_clean else ""

        # Build vocabulary of substantive words (3+ alpha characters)
        vocab_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", t_clean.lower()))

        is_strict = STRICT_TRANSCRIPT_ONLY if strict_mode is None else bool(strict_mode)

        if domain is None:
            # Auto-classify (module-level function defined below)
            detected_domain, detected_conf = classify_domain(t_clean)
        else:
            detected_domain = domain
            detected_conf = confidence if confidence is not None else 1.0

        return cls(
            recording_id=recording_id,
            transcript_hash=t_hash,
            domain=detected_domain,
            confidence=detected_conf,
            vocab=vocab_words,
            strict_mode=is_strict,
        )

    def get_rng(self, salt: str = "") -> random.Random:
        """
        Return an isolated, deterministic random.Random instance seeded by
        recording_id, transcript_hash, and optional salt.
        Replaces global random.seed().
        """
        seed_str = f"{self.recording_id}:{self.transcript_hash[:16]}:{salt}"
        seed_int = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest()[:8], 16)
        return random.Random(seed_int)

    def is_word_in_transcript(self, word: str) -> bool:
        """Check if a single word or phrase appears in the recording vocabulary."""
        w_low = word.lower().strip()
        if not w_low:
            return False
        if " " in w_low or "-" in w_low:
            parts = re.findall(r"\b[a-zA-Z]{3,}\b", w_low)
            return all(p in self.vocab for p in parts)
        return w_low in self.vocab


# ---------------------------------------------------------------------------
# Domain Gate
# ---------------------------------------------------------------------------

# Weighted term lists for domain classification.
# Ambiguous terms (model, learning, profit, queue, loss, accuracy, etc.) have
# near-zero weight and are deliberately excluded.
_DOMAIN_TERM_WEIGHTS: dict[str, dict[str, float]] = {
    "operating_system": {
        "process": 1.0, "cpu": 2.0, "thread": 1.5, "scheduling": 2.0,
        "deadlock": 2.0, "semaphore": 2.0, "mutex": 2.0, "preemptive": 2.0,
        "fcfs": 3.0, "convoy": 3.0, "turnaround": 2.0, "burst": 1.5,
        "ready": 0.5, "dispatcher": 2.0, "context": 0.3, "pcb": 3.0,
        "round": 0.3, "quantum": 1.0, "starvation": 1.5, "aging": 1.0,
    },
    "machine_learning": {
        "supervised": 2.5, "unsupervised": 2.5, "classification": 1.5,
        "regression": 1.5, "overfitting": 2.5, "underfitting": 2.5,
        "gradient": 2.0, "backpropagation": 3.0, "neural": 1.5,
        "dataset": 1.5, "labeled": 2.0, "features": 1.0, "training": 1.0,
        "epoch": 2.0, "hyperparameter": 2.5, "regularization": 2.0,
    },
    "accounting_finance": {
        "accrual": 3.0, "depreciation": 2.5, "revenue": 1.5,
        "expenditure": 2.0, "balance": 1.0, "ledger": 3.0,
        "debit": 3.0, "credit": 1.0, "invoice": 2.5, "liability": 2.0,
        "asset": 1.5, "equity": 2.0, "receivable": 2.5, "payable": 2.5,
        "liquidity": 2.0, "insolvency": 3.0,
    },
    "fintech": {
        "fintech": 3.0, "digital": 0.5, "blockchain": 3.0,
        "cryptocurrency": 3.0, "wallet": 1.5, "payment": 1.5,
        "kyc": 3.0, "robo": 3.0, "insurtech": 3.0, "regtech": 3.0,
        "neobank": 3.0, "underwriting": 2.5, "fraud": 1.5,
    },
    "biology": {
        "neuron": 2.5, "synapse": 3.0, "reflex": 2.5, "stimulus": 2.0,
        "receptor": 1.5, "hormone": 2.5, "endocrine": 3.0, "dna": 3.0,
        "chromosome": 3.0, "mitosis": 3.0, "photosynthesis": 3.0,
        "enzyme": 2.5, "protein": 1.5, "cell": 1.0, "organism": 1.5,
        "nervous": 2.0, "spinal": 2.5, "coordination": 1.0,
    },
    "physics": {
        "voltage": 2.5, "current": 1.0, "resistance": 1.5, "ohm": 3.0,
        "coulomb": 3.0, "ampere": 2.5, "watt": 2.5, "joule": 2.5,
        "capacitor": 3.0, "inductor": 3.0, "circuit": 2.0, "electron": 2.5,
        "conductor": 2.0, "insulator": 2.0, "resistivity": 3.0,
        "potential": 1.0, "electromagnetic": 3.0,
    },
}

_DOMAIN_MIN_DISTINCT_TERMS = 4     # minimum distinct matching terms
_DOMAIN_MIN_TOTAL_WEIGHT = 5.0     # minimum cumulative weight
_DOMAIN_MARGIN_RATIO = 1.5         # winner must score >= 1.5× the runner-up


def classify_domain(transcript_text: str) -> tuple[str, float]:
    """
    Classify the domain of a transcript using weighted term matching.

    Returns (domain_label, confidence) where domain_label is one of the
    _DOMAIN_TERM_WEIGHTS keys or 'general', and confidence is in [0, 1].

    A domain passes only when:
    - At least _DOMAIN_MIN_DISTINCT_TERMS distinct terms matched
    - Total weight >= _DOMAIN_MIN_TOTAL_WEIGHT
    - Winner score >= _DOMAIN_MARGIN_RATIO * runner-up score
    """
    text_low = (transcript_text or "").lower()
    # Tokenise to word-boundary set for O(1) lookup
    text_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", text_low))

    scores: dict[str, float] = {}
    distinct_hits: dict[str, int] = {}

    for domain, term_weights in _DOMAIN_TERM_WEIGHTS.items():
        total_w = 0.0
        n_hits = 0
        for term, weight in term_weights.items():
            # Support multi-word terms (e.g. "round robin")
            if " " in term:
                if term in text_low:
                    total_w += weight
                    n_hits += 1
            else:
                if term in text_tokens:
                    total_w += weight
                    n_hits += 1
        scores[domain] = total_w
        distinct_hits[domain] = n_hits

    if not scores:
        return "general", 0.0

    sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winner_domain, winner_score = sorted_domains[0]
    runner_up_score = sorted_domains[1][1] if len(sorted_domains) > 1 else 0.0

    passes_min_terms = distinct_hits.get(winner_domain, 0) >= _DOMAIN_MIN_DISTINCT_TERMS
    passes_min_weight = winner_score >= _DOMAIN_MIN_TOTAL_WEIGHT
    passes_margin = (runner_up_score == 0.0) or (winner_score >= _DOMAIN_MARGIN_RATIO * runner_up_score)

    if passes_min_terms and passes_min_weight and passes_margin:
        # Normalise confidence to [0, 1]
        max_possible = sum(_DOMAIN_TERM_WEIGHTS[winner_domain].values())
        confidence = min(1.0, winner_score / max_possible)
        return winner_domain, round(confidence, 3)

    return "general", 0.0
