"""
app/services/contextual_transcription.py — Context-Aware Semantic Transcription Refiner.

Problem:
    Whisper decodes audio acoustically, not semantically. In bilingual CS lectures
    (English + Roman Urdu), acoustic phonemes frequently map to wrong words:
      - "threat" instead of "thread"
      - "computer and time written main" instead of "compute time and waiting time"
      - "paging can reduce starvation" instead of "aging can reduce starvation"
      - "a project knowledge" instead of "a priori knowledge"
      - garbled Roman Urdu code-switches

    The existing domain_corrector.py handles token-level and 2-gram phonetic fixes,
    but it lacks sentence-level semantic context awareness.

Solution:
    This module adds a *contextual refinement layer* that operates on full segments
    with awareness of surrounding segments (sliding window). It:
      1. Auto-detects the lecture domain (OS, DS, Networks, ML, etc.).
      2. Applies context-dependent collocation rules that require surrounding words
         to disambiguate (e.g., "paging" -> "aging" only near "starvation" + "reduce").
      3. Repairs garbled bilingual code-switched phrases.
      4. Prunes hallucinated noise segments.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from app.config import HALLUCINATION_TOKEN_RATIO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain Detection
# ---------------------------------------------------------------------------
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "operating_systems": [
        "process", "scheduling", "cpu burst", "round robin", "fcfs", "sjf", "srtf",
        "context switch", "ready queue", "waiting time", "turnaround time",
        "deadlock", "semaphore", "mutex", "kernel", "virtual memory",
        "page fault", "thrashing", "preemptive", "non-preemptive", "dispatcher",
        "gantt chart", "convoy effect", "starvation", "aging", "time quantum",
        "process synchronization", "critical section", "race condition",
        "priority scheduling", "operating system",
    ],
    "data_structures": [
        "binary search tree", "avl tree", "linked list", "stack", "queue",
        "heap", "hash table", "graph", "traversal", "breadth first",
        "depth first", "sorting", "quicksort", "mergesort", "big o",
        "time complexity", "space complexity", "recursion", "dynamic programming",
    ],
    "networks": [
        "tcp", "udp", "ip address", "osi model", "packet", "routing",
        "socket", "bandwidth", "latency", "dns", "http", "https",
        "handshake", "congestion", "flow control", "subnet",
    ],
    "machine_learning": [
        "neural network", "deep learning", "gradient descent", "backpropagation",
        "loss function", "activation function", "supervised", "unsupervised",
        "overfitting", "regularization", "transformer", "attention",
        "convolutional", "recurrent", "lstm", "classification", "regression",
    ],
    "databases": [
        "relational database", "normalization", "sql", "primary key",
        "foreign key", "join", "transaction", "acid", "indexing",
        "query optimization", "functional dependency", "candidate key",
    ],
}


def detect_lecture_domain(transcript_text: str) -> str:
    """
    Auto-detect the primary lecture domain from transcript content.
    Returns the domain key with the highest keyword hit count.
    Falls back to 'general_cs' if no domain scores highly.
    """
    text_lower = transcript_text.lower()
    scores: dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[domain] = score

    if not scores:
        return "general_cs"

    best_domain = max(scores, key=lambda d: scores[d])
    if scores[best_domain] < 3:
        return "general_cs"
    return best_domain


# ---------------------------------------------------------------------------
# Contextual Collocation Rules
# ---------------------------------------------------------------------------

@dataclass
class ContextualRule:
    """A context-sensitive replacement rule."""
    pattern: re.Pattern
    replacement: str
    context_required: list[str] = field(default_factory=list)
    context_forbidden: list[str] = field(default_factory=list)
    domain_required: list[str] = field(default_factory=list)

    def matches_context(self, context: str, domain: str) -> bool:
        ctx_lower = context.lower()
        if self.context_required and not any(c in ctx_lower for c in self.context_required):
            return False
        if self.context_forbidden and any(c in ctx_lower for c in self.context_forbidden):
            return False
        if self.domain_required and domain not in self.domain_required and "general_cs" not in self.domain_required:
            return False
        return True


CONTEXTUAL_COLLOCATION_RULES: list[ContextualRule] = [
    # --- Homophone Disambiguation ---

    # "paging can reduce starvation" -> "aging can reduce starvation"
    ContextualRule(
        pattern=re.compile(r'\bpaging\b', re.IGNORECASE),
        replacement="aging",
        context_required=["starvation", "reduce", "starv"],
        domain_required=["operating_systems", "general_cs"],
    ),

    # "threat" -> "thread" in process/program context
    ContextualRule(
        pattern=re.compile(r'\bthreat\b', re.IGNORECASE),
        replacement="thread",
        context_required=["process", "program", "cpu", "scheduling", "multithreading"],
    ),
    ContextualRule(
        pattern=re.compile(r'\bthreats\b', re.IGNORECASE),
        replacement="threads",
        context_required=["process", "program", "cpu", "scheduling", "multithreading"],
    ),

    # "a project knowledge" -> "a priori knowledge"
    ContextualRule(
        pattern=re.compile(r'\ba\s+project\s+knowledge\b', re.IGNORECASE),
        replacement="a priori knowledge",
        context_required=["predict", "cpu burst", "scheduling", "sjf", "future"],
    ),

    # "a side of priority" -> "assigned priority"
    ContextualRule(
        pattern=re.compile(r'\ba\s+side\s+of\s+priority\b', re.IGNORECASE),
        replacement="assigned priority",
        context_required=["scheduling", "priority", "process"],
    ),

    # "SGF non-cleanable" -> "SJF non-preemptive"
    ContextualRule(
        pattern=re.compile(r'\bSGF\s+non[- ]?clean(?:able|ible)\b', re.IGNORECASE),
        replacement="SJF non-preemptive",
        context_required=[],
        domain_required=["operating_systems", "general_cs"],
    ),

    # "one-size process" -> "once a process"
    ContextualRule(
        pattern=re.compile(r'\bone[- ]size\s+process\b', re.IGNORECASE),
        replacement="once a process",
        context_required=["cpu", "sjf", "non-preemptive", "scheduling", "finish"],
    ),

    # "rival time" -> "arrival time"
    ContextualRule(
        pattern=re.compile(r'\brival\s+time\b', re.IGNORECASE),
        replacement="arrival time",
        context_required=["completion", "turnaround", "process", "burst", "scheduling"],
    ),

    # "first time 2" -> "arrival time 2" (in scheduling numerical context)
    ContextualRule(
        pattern=re.compile(r'\bfirst\s+time\s+(\d)\b', re.IGNORECASE),
        replacement=r"arrival time \1",
        context_required=["completion", "turnaround", "burst"],
    ),

    # "feature line" -> "queue" (metaphorical)
    ContextualRule(
        pattern=re.compile(r'\bfeature\s+line\b', re.IGNORECASE),
        replacement="queue",
        context_required=["convoy", "stuck", "stack", "process", "waiting"],
    ),

    # "X load truck" -> "a slow truck" (convoy effect metaphor)
    ContextualRule(
        pattern=re.compile(r'\bX\s+load\s+truck\b', re.IGNORECASE),
        replacement="a slow truck",
        context_required=["motorway", "convoy", "line", "stack"],
    ),

    # "is stack" -> "is stuck" (Urdu-English confusion)
    ContextualRule(
        pattern=re.compile(r'\bis\s+stack\b', re.IGNORECASE),
        replacement="is stuck",
        context_required=["line", "queue", "convoy", "process", "waiting"],
    ),

    # "deletive, da, starvation, gation" -> "relatively, and starvation"
    ContextualRule(
        pattern=re.compile(r'\bdeletive,?\s*da,?\s*starvation,?\s*gation\b', re.IGNORECASE),
        replacement="relatively, and starvation",
        context_required=["fcfs", "advantage", "simple"],
    ),

    # "the purity" -> "priority"
    ContextualRule(
        pattern=re.compile(r'\bthe\s+purity\b', re.IGNORECASE),
        replacement="priority",
        context_required=["scheduling", "process", "priority"],
    ),

    # "post-mone" -> "postponement"
    ContextualRule(
        pattern=re.compile(r'\bpost[- ]?mone\b', re.IGNORECASE),
        replacement="postponement",
        context_required=["starvation", "indefinite", "wait"],
    ),

    # "sockets danray" -> "sochte hain ke"
    ContextualRule(
        pattern=re.compile(r'\bsockets\s+danray\b', re.IGNORECASE),
        replacement="sochte hain ke",
        context_required=["starvation", "wait", "process"],
    ),

    # "for data development to be reduced" -> "for usko prevent karne ke liye"
    ContextualRule(
        pattern=re.compile(r'\bfor\s+data\s+development\s+to\s+be\s+reduced\b', re.IGNORECASE),
        replacement="for usko prevent karne ke liye",
        context_required=["starvation", "aging"],
    ),

    # "different subtl" -> "different, subtle"
    ContextualRule(
        pattern=re.compile(r'\bdifferent\s+subtl\b', re.IGNORECASE),
        replacement="different, subtle",
        context_required=["srtf", "sjf", "shortest"],
    ),

    # "and laptop" -> "and last process" (scheduling context)
    ContextualRule(
        pattern=re.compile(r'\band\s+laptop\b', re.IGNORECASE),
        replacement="and last process",
        context_required=["ready queue", "cpu burst", "scheduler"],
        context_forbidden=["computer", "device"],
    ),

    # "deep any application" -> "keep any application"
    ContextualRule(
        pattern=re.compile(r'\bdeep\s+any\s+application\b', re.IGNORECASE),
        replacement="keep any application",
        context_required=["program", "executable", "file", "disk"],
    ),

    # "Likhe" -> "Like" when followed by English technical context
    ContextualRule(
        pattern=re.compile(r'\bLikhe\b'),
        replacement="Like",
        context_required=["disk", "io", "cpu", "process", "compute"],
    ),

    # "computer and time" -> "compute time, and"
    ContextualRule(
        pattern=re.compile(r'\bcomputer\s+and\s+time\b', re.IGNORECASE),
        replacement="compute time, and",
        context_required=["disk", "io", "process", "waiting", "finished"],
    ),

    # "written main process" -> "waiting mein process"
    ContextualRule(
        pattern=re.compile(r'\bwritten\s+main\s+process\b', re.IGNORECASE),
        replacement="waiting mein process",
        context_required=["disk", "io", "cpu", "finished", "compute"],
    ),

    # "Here/Mere is a common confusion" -> "Yahan ek common confusion hai"
    ContextualRule(
        pattern=re.compile(r'\b[MH]ere\s+is\s+a\s+common\s+confusion\b', re.IGNORECASE),
        replacement="Yahan ek common confusion hai",
        context_required=["ready", "cpu", "process", "waiting"],
    ),

    # "Rode map" -> "Road map"
    ContextualRule(
        pattern=re.compile(r'\bRode\s+map\b', re.IGNORECASE),
        replacement="Road map",
        context_required=[],
    ),

    # "milli seps" -> "milliseconds"
    ContextualRule(
        pattern=re.compile(r'\bmilli?\s*seps?\b', re.IGNORECASE),
        replacement="milliseconds",
        context_required=[],
    ),

    # "1 more cheese" -> "Ek ahem cheez"
    ContextualRule(
        pattern=re.compile(r'\b1\s+more\s+cheese\b', re.IGNORECASE),
        replacement="Ek ahem cheez",
        context_required=[],
    ),

    # "cut token" -> "karein to"
    ContextualRule(
        pattern=re.compile(r'\bcut\s+token\b', re.IGNORECASE),
        replacement="karein to",
        context_required=["process", "queue", "line"],
    ),

    # "Banda, Pelle, normal use" -> "Banda, pehle normal use"
    ContextualRule(
        pattern=re.compile(r'\bBanda,?\s*Pelle,?\s*normal\s+use\b', re.IGNORECASE),
        replacement="Banda, pehle normal use",
        context_required=["process", "queue", "fcfs"],
    ),

    # "Pelle process" -> "pehle process"
    ContextualRule(
        pattern=re.compile(r'\bPelle\s+process\b', re.IGNORECASE),
        replacement="pehle process",
        context_required=["queue", "line", "cpu"],
    ),

    # "Key end may be" -> "Ke end mein"
    ContextualRule(
        pattern=re.compile(r'\bKey?\s+end\s+ma[yt]\s+be\b', re.IGNORECASE),
        replacement="Ke end mein",
        context_required=["p1", "p2", "p3", "time quantum", "round robin"],
    ),

    # "in definite post-ponement" -> "indefinite postponement"
    ContextualRule(
        pattern=re.compile(r'\bin\s+definite\s+post[- ]?ponement\b', re.IGNORECASE),
        replacement="indefinite postponement",
        context_required=["starvation", "sjf", "preemptive"],
    ),

    # "the par is" -> "the car is" (convoy metaphor)
    ContextualRule(
        pattern=re.compile(r'\bthe\s+par\s+is\b', re.IGNORECASE),
        replacement="the car is",
        context_required=["motorway", "truck", "convoy", "speed"],
    ),

    # "neend to" -> "need to"
    ContextualRule(
        pattern=re.compile(r'\bneend\s+to\b', re.IGNORECASE),
        replacement="need to",
        context_required=[],
    ),

    # "hai passive collection" -> "hai ek passive collection"
    ContextualRule(
        pattern=re.compile(r'\bhai\s+passive\s+collection\b', re.IGNORECASE),
        replacement="hai ek passive collection",
        context_required=["program", "instructions"],
    ),

    # "does not mein" -> "does not mean"
    ContextualRule(
        pattern=re.compile(r'\bdoes\s+not\s+mein\b', re.IGNORECASE),
        replacement="does not mean",
        context_required=["cpu", "process", "automatically"],
    ),

    # "tha process/is/all/algorithm" -> "that ..."
    ContextualRule(
        pattern=re.compile(r'\btha\s+(process|is|all|algorithm)\b', re.IGNORECASE),
        replacement=r"that \1",
        context_required=[],
    ),

    # "thi is" -> "this is"
    ContextualRule(
        pattern=re.compile(r'\bthi\s+is\b', re.IGNORECASE),
        replacement="this is",
        context_required=[],
    ),

    # "the why not" -> "then why not"
    ContextualRule(
        pattern=re.compile(r'\bthe\s+why\s+not\b', re.IGNORECASE),
        replacement="then why not",
        context_required=["ready", "cpu", "process"],
    ),

    # "to different windows" -> "two different windows"
    ContextualRule(
        pattern=re.compile(r'\bto\s+different\s+windows\b', re.IGNORECASE),
        replacement="two different windows",
        context_required=["instances", "process", "multiple"],
    ),

    # "the p3 first" -> "then p3 first" (scheduling)
    ContextualRule(
        pattern=re.compile(r'\bthe\s+(p[1-9])\s+first\b', re.IGNORECASE),
        replacement=r"then \1 first",
        context_required=["burst", "ready", "sjf", "scheduling"],
    ),

    # "refleccion" -> "reflection"
    ContextualRule(
        pattern=re.compile(r'\brefleccion\b', re.IGNORECASE),
        replacement="reflection",
        context_required=[],
    ),

    # "shwanda" -> "" (noise)
    ContextualRule(
        pattern=re.compile(r'\bshwanda\b', re.IGNORECASE),
        replacement="",
        context_required=[],
    ),

    # "Baba Chamakthak" -> "" (noise)
    ContextualRule(
        pattern=re.compile(r'\bBaba\s+Chamakthak\b', re.IGNORECASE),
        replacement="",
        context_required=[],
    ),
]


# ---------------------------------------------------------------------------
# Garbled Roman Urdu Code-Switch Repair
# ---------------------------------------------------------------------------
GARBLED_PHRASE_REPAIRS: list[tuple[re.Pattern, str]] = [
    # "skajilad kohe kpallus si follow kurni ogin" -> "scheduler ko ek policy follow karni hogi"
    (re.compile(r'skajilad\s+kohe?\s+kpallus\s+si\s+follow\s+kurni\s+ogin', re.IGNORECASE),
     "scheduler ko ek policy follow karni hogi"),

    # "t1-10 post-mone, sockets danray" -> "indefinite postponement, sochte hain ke"
    (re.compile(r't1-10\s+post-mone,?\s*sockets\s+danray', re.IGNORECASE),
     "indefinite postponement, sochte hain ke"),

    # "for jata development to be reduced, aaching, ageing" -> "for usko reduce karne ke liye, aging"
    (re.compile(r'for\s+jata\s+development\s+to\s+be\s+reduced,?\s*aaching,?\s*ageing', re.IGNORECASE),
     "for usko reduce karne ke liye, aging"),

    # "literally whatever wah to wait can be" -> "literally whatever way to wait, it can be"
    (re.compile(r'literally\s+whatever\s+wah\s+to\s+wait\s+can\s+be\b', re.IGNORECASE),
     "literally whatever way to wait, it can be"),

    # "coach interactive/process/background" -> "kuch ..."
    (re.compile(r'\bcoach\s+(interactive|process|background)', re.IGNORECASE),
     r"kuch \1"),
]


# ---------------------------------------------------------------------------
# Hallucination / Noise Segment Detection
# ---------------------------------------------------------------------------
NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(r'^\s*$'),                                            # Empty
    re.compile(r'^[a-z]{1,3}$', re.IGNORECASE),                    # 1-3 letter fragments: "se", "re", "as"
    re.compile(r'^dhk$', re.IGNORECASE),                            # Common Whisper noise token
    re.compile(r'^bk\s', re.IGNORECASE),                            # "bk ..." noise
    re.compile(r'^ske$', re.IGNORECASE),                            # isolated "ske"
    re.compile(r'^Baba\s+Chamakthak$', re.IGNORECASE),              # Whisper hallucination
    re.compile(r'^ala$', re.IGNORECASE),                            # Isolated fragment
    re.compile(r'^(?:mhebi|mhmi|diphthim|shwanda|invincible|bred)$', re.IGNORECASE), # Acoustic hallucinations
    re.compile(r'[\u0400-\u04FF]'),                                 # Cyrillic characters (noise)
    re.compile(r'[\u0B80-\u0BFF]'),                                 # Tamil characters (noise)
    re.compile(r'[\u0900-\u097F]'),                                 # Devanagari (noise)
    re.compile(r'[\u4E00-\u9FFF]'),                                 # CJK (noise)
]

_HALLUCINATION_NOISE_TOKENS = {
    "mhebi", "mhmi", "diphthim", "dhk", "ske", "ala", "chamakthak", "shwanda"
}


def is_noise_segment(text: str, duration: float = 0.0) -> bool:
    """
    Determine whether a segment is hallucinated noise that should be pruned.
    """
    if not text or not text.strip():
        return True

    cleaned = text.strip()

    # Check against known noise patterns
    for pat in NOISE_PATTERNS:
        if pat.search(cleaned):
            return True

    words = [w.lower() for w in re.findall(r'[A-Za-z0-9]+', cleaned)]
    if not words:
        return True

    # Check if segment consists predominantly of acoustic hallucination tokens
    h_count = sum(1 for w in words if w in _HALLUCINATION_NOISE_TOKENS)
    if h_count > 0 and (h_count / len(words) >= HALLUCINATION_TOKEN_RATIO or len(words) <= 3):
        return True

    # Single isolated non-technical words across short/medium spans
    technical_keepers = {
        "gantt chart", "round robin", "fcfs", "sjf", "srtf",
        "convoy effect", "race condition", "critical section",
        "process synchronization", "formulas", "chart", "next lecture",
        "thank you", "template", "portal", "submission", "submit",
        "deadline", "assignment", "verify", "due", "process", "cpu",
        "burst", "waiting", "turnaround", "scheduling", "preemptive"
    }
    cleaned_lower = cleaned.lower()

    if len(words) == 1 and not any(tk in cleaned_lower for tk in technical_keepers):
        if duration < 5.0:
            return True

    # Very short segments (<0.8s) with very few words and no technical/academic content
    if duration > 0 and duration < 0.8:
        if len(words) <= 2:
            if not any(tk in cleaned_lower for tk in technical_keepers):
                return True

    return False


# ---------------------------------------------------------------------------
# Core Contextual Refinement Engine
# ---------------------------------------------------------------------------
@dataclass
class RefinedSegment:
    """A refined transcript segment."""
    segment_index: int
    original_text: str
    refined_text: str
    was_pruned: bool = False
    changes_made: list[str] = field(default_factory=list)


class ContextualTranscriptionRefiner:
    """
    Context-aware semantic transcription refiner.

    Operates on segment batches with a sliding context window.
    Uses rule-based contextual disambiguation — no LLM required.
    """

    def __init__(self, domain: Optional[str] = None):
        self.domain = domain or "general_cs"

    def detect_and_set_domain(self, full_transcript: str) -> str:
        """Auto-detect and set lecture domain from full transcript text."""
        self.domain = detect_lecture_domain(full_transcript)
        logger.info(f"Detected lecture domain: {self.domain}")
        return self.domain

    def refine_segments(
        self,
        segments: list[dict],
        context_window: int = 3,
    ) -> list[RefinedSegment]:
        """
        Refine all segments with contextual awareness.

        Args:
            segments: List of dicts with keys: segment_index, text, start_time, end_time
            context_window: Number of surrounding segments to include in context

        Returns:
            List of RefinedSegment with refined text and change annotations.
        """
        results: list[RefinedSegment] = []

        for i, seg in enumerate(segments):
            seg_text = seg.get("text", "").strip()
            seg_idx = seg.get("segment_index", i)
            start_time = seg.get("start_time", 0.0)
            end_time = seg.get("end_time", 0.0)
            duration = end_time - start_time

            # Step 1: Check for noise/hallucination
            if is_noise_segment(seg_text, duration):
                results.append(RefinedSegment(
                    segment_index=seg_idx,
                    original_text=seg_text,
                    refined_text="",
                    was_pruned=True,
                    changes_made=["pruned: hallucinated noise segment"],
                ))
                continue

            # Step 2: Build context window from surrounding segments
            context_parts = []
            for j in range(max(0, i - context_window), min(len(segments), i + context_window + 1)):
                if j != i:
                    neighbor_text = segments[j].get("text", "")
                    if neighbor_text and not is_noise_segment(neighbor_text, 0):
                        context_parts.append(neighbor_text)
            context_str = " ".join(context_parts)

            # Step 3: Apply garbled phrase repairs first (full-match patterns)
            refined = seg_text
            changes = []

            for pattern, replacement in GARBLED_PHRASE_REPAIRS:
                new_text = pattern.sub(replacement, refined)
                if new_text != refined:
                    changes.append(f"garbled_repair: matched")
                    refined = new_text

            # Step 4: Apply contextual collocation rules
            for rule in CONTEXTUAL_COLLOCATION_RULES:
                if rule.matches_context(context_str + " " + refined, self.domain):
                    new_text = rule.pattern.sub(rule.replacement, refined)
                    if new_text != refined:
                        changes.append(f"context_rule: '{rule.replacement}'")
                        refined = new_text

            # Step 5: Clean up spacing and punctuation artifacts
            refined = re.sub(r'\s{2,}', ' ', refined).strip()
            refined = re.sub(r'\s+([,.])', r'\1', refined)

            # Step 6: If the result is now empty after repairs, prune
            if not refined or len(refined.strip()) < 2:
                results.append(RefinedSegment(
                    segment_index=seg_idx,
                    original_text=seg_text,
                    refined_text="",
                    was_pruned=True,
                    changes_made=["pruned: empty after refinement"],
                ))
                continue

            results.append(RefinedSegment(
                segment_index=seg_idx,
                original_text=seg_text,
                refined_text=refined,
                was_pruned=False,
                changes_made=changes,
            ))

        return results

    def refine_single_segment(
        self,
        text: str,
        context_before: str = "",
        context_after: str = "",
    ) -> str:
        """
        Refine a single segment's text with optional before/after context.
        Returns the refined text.
        """
        seg = {"segment_index": 0, "text": text, "start_time": 0, "end_time": 10}
        context_segs = []
        if context_before:
            context_segs.append({"segment_index": -1, "text": context_before, "start_time": 0, "end_time": 5})
        context_segs.append(seg)
        if context_after:
            context_segs.append({"segment_index": 1, "text": context_after, "start_time": 10, "end_time": 15})

        results = self.refine_segments(context_segs)
        for r in results:
            if r.segment_index == 0:
                return r.refined_text if not r.was_pruned else ""
        return text


def refine_transcript_segments(
    segments: list[dict],
    full_transcript: str = "",
) -> list[RefinedSegment]:
    """
    Top-level convenience function: refine a list of transcript segments.

    Args:
        segments: List of dicts with keys: segment_index, text, start_time, end_time
        full_transcript: Optional full transcript text for domain detection

    Returns:
        List of RefinedSegment with contextually refined text.
    """
    refiner = ContextualTranscriptionRefiner()

    # Auto-detect domain
    if full_transcript:
        refiner.detect_and_set_domain(full_transcript)
    elif segments:
        combined = " ".join(s.get("text", "") for s in segments)
        refiner.detect_and_set_domain(combined)

    return refiner.refine_segments(segments)
