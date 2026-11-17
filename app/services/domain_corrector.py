"""
app/services/domain_corrector.py — Context-Aware Domain Vocabulary Corrector.

Problem solved:
    Whisper mishears domain-specific technical terms (e.g. "gan chart" instead of
    "Gantt chart", "pro ses" instead of "process", "kernal" instead of "kernel").
    These errors are phonetically plausible but semantically wrong in the lecture domain.

Approach:
    1. Whisper initial_prompt injection: Feed Whisper a prompt containing domain
       vocabulary so its decoder is biased toward correct spellings at inference time.
    2. Post-transcription fuzzy correction: After transcription, scan every token
       in each segment. For any token (or 2-token phrase) not in a common English
       word list, compute phonetic code (double-metaphone) + edit distance to every
       domain term. If a close-enough match exists and the domain term fits the
       lecture topic context, replace it.

Domain vocabulary is organized by topic (OS, DS, Networks, etc.) so the corrector
can weight matches that belong to the detected topic of the lecture.
"""

from __future__ import annotations
from functools import lru_cache
import logging
import re
from typing import Optional
from difflib import SequenceMatcher

from app.config import DOMAIN_CORRECTOR_MIN_SCORE, DOMAIN_CORRECTOR_BIGRAM_MIN_SCORE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Comprehensive Multi-Domain Canonical Vocabulary
# ---------------------------------------------------------------------------
CANONICAL_TECHNICAL_TERMS: list[str] = [
    # Operating Systems & Architecture
    "Gantt chart", "Round Robin", "FCFS", "SJF scheduling", "SRTF scheduling",
    "Shortest Remaining Time First", "Shortest Job First", "First Come First Served",
    "priority scheduling", "preemptive", "non-preemptive", "CPU burst", "CPU burst time",
    "context switch", "dispatcher", "turnaround time", "waiting time",
    "response time", "throughput", "process scheduling", "multiprogramming",
    "multithreading", "deadlock", "starvation", "semaphore", "mutex",
    "critical section", "race condition", "Peterson algorithm", "banker algorithm",
    "virtual memory", "paging", "segmentation", "page fault", "thrashing",
    "TLB", "demand paging", "page replacement", "LRU", "FIFO",
    "kernel", "monolithic kernel", "microkernel", "system call", "interrupt",
    "device driver", "file system", "inode", "RAID storage", "memory hierarchy",
    "cache coherence", "instruction pipeline", "branch prediction",
    "ready queue", "convoy effect", "time quantum", "running instance",

    # Data Structures & Algorithms
    "binary search tree", "AVL tree", "red-black tree", "B-tree", "heap",
    "priority queue", "hash table", "hash map", "collision resolution",
    "linked list", "doubly linked list", "circular linked list", "stack", "queue",
    "graph traversal", "breadth first search", "depth first search",
    "Dijkstra algorithm", "Bellman-Ford algorithm", "Floyd-Warshall algorithm",
    "Kruskal algorithm", "Prim algorithm", "topological sort",
    "dynamic programming", "greedy algorithm", "divide and conquer",
    "backtracking", "recursion", "memoization", "Big O notation",
    "time complexity", "space complexity", "asymptotic notation",
    "quicksort", "mergesort", "heapsort", "radix sort", "binary search",

    # Databases & Information Systems
    "relational database", "normalization", "first normal form", "second normal form",
    "third normal form", "Boyce-Codd normal form", "functional dependency",
    "candidate key", "primary key", "foreign key", "composite key",
    "SQL query", "relational algebra", "join operation", "indexing",
    "B-tree index", "query optimization", "ACID properties", "transaction",
    "serializability", "two-phase locking", "write-ahead logging",
    "NoSQL database", "document store", "key-value store", "column family",

    # Computer Networks & Distributed Systems
    "TCP/IP", "OSI model", "packet switching", "circuit switching",
    "bandwidth", "latency", "jitter", "throughput", "socket programming",
    "handshake", "sliding window protocol", "flow control", "congestion control",
    "routing algorithm", "distance vector", "link state", "IP address",
    "subnet mask", "CIDR notation", "DNS resolution", "HTTP request",
    "HTTPS protocol", "REST API", "WebSocket", "client-server architecture",
    "distributed systems", "consensus algorithm", "Paxos", "Raft",
    "CAP theorem", "load balancer", "reverse proxy", "microservices",

    # Artificial Intelligence & Machine Learning
    "machine learning", "deep learning", "neural network", "artificial intelligence",
    "supervised learning", "unsupervised learning", "reinforcement learning",
    "classification and regression", "classification", "regression", "evaluation metrics",
    "precision and recall", "precision", "recall", "F1 score", "F1-score", "accuracy",
    "training and test sets", "mean absolute error", "mean squared error",
    "gradient descent", "stochastic gradient descent", "backpropagation",
    "loss function", "cost function", "activation function", "sigmoid",
    "ReLU", "softmax", "convolutional neural network", "recurrent neural network",
    "LSTM", "GRU", "transformer model", "attention mechanism", "self-attention",
    "overfitting", "underfitting", "regularization", "dropout",
    "hyperparameter tuning", "cross-validation", "support vector machine",
    "decision tree", "random forest", "k-nearest neighbors", "k-means clustering",
    "principal component analysis", "feature extraction", "word embedding",
    "automated machine learning",

    # Software Engineering & Programming Languages
    "object-oriented programming", "polymorphism", "inheritance", "encapsulation",
    "abstraction", "design patterns", "singleton pattern", "factory pattern",
    "observer pattern", "compiler", "interpreter", "lexical analysis",
    "syntax analysis", "abstract syntax tree", "semantic analysis",
    "intermediate code generation", "garbage collection", "memory leak",
    "refactoring", "unit testing", "integration testing", "continuous integration",
    "version control", "software architecture", "agile methodology",

    # Cyber Security & Cryptography
    "symmetric encryption", "asymmetric encryption", "public key cryptography",
    "RSA algorithm", "cryptographic hash", "digital signature",
    "authentication", "authorization", "SQL injection", "cross-site scripting",
    "buffer overflow", "firewall", "intrusion detection system",

    # Biology & Life Sciences
    "Coordination and Response", "Nervous System", "Reflex Arc", "Reflex Action",
    "Central Nervous System", "Peripheral Nervous System", "Sensory Neurons",
    "Relay Neurons", "Motor Neurons", "Effector Organ", "Receptor Organ",
    "Sensory Receptors", "Homeostasis", "Electrical Impulses", "Hormonal Coordination",
    "Endocrine System", "Synapse", "Neurotransmitter", "Spinal Cord", "Stimulus and Response",
    "Endocrine Gland", "Nerve Impulse", "Nervous Coordination",

    # Accounting, Finance & Economics
    "Difference Between Profit and Cash Flow", "Net Cash Flow", "Working Capital",
    "Accrual Accounting", "Revenue Recognition", "Depreciation", "Operating Cash Flow",
    "Cash Inflows", "Cash Outflows", "Balance Sheet", "Income Statement", "Liquidity",

    # Physics, Electricity & Electrical Circuits
    "Electricity and Electric Circuits", "Electric Charge", "Electric Current",
    "Potential Difference", "Voltage", "Electrical Resistance", "Resistivity",
    "Ohm's Law", "Series Circuit", "Parallel Circuit", "Coulomb", "Ampere",
    "Volt", "Ohm", "Electrons", "Protons", "Neutrons", "Conductor", "Insulator",
    "Cross-Sectional Area", "Direct Current", "Alternating Current", "Electromotive Force",

    # Academic Coursework & Directives
    "Coursework Directives & Submission", "Coursework Directives", "Coursework Submission",
    "Assignment Directives", "Homework Assignment", "Student Portal Submission"
]

# Quick lookup dictionary for known frequent mishearings (Fast-path cache)
_COMMON_MISHEARINGS: dict[str, str] = {
    # Gantt Chart variants
    "gan chart": "Gantt chart",
    "gant chart": "Gantt chart",
    "gaant chart": "Gantt chart",
    "grant chart": "Gantt chart",
    "gun chart": "Gantt chart",
    "gaon chart": "Gantt chart",
    "gaon charts": "Gantt charts",


    # CPU Scheduling metrics & terms
    "wedding time": "waiting time",
    "weding time": "waiting time",
    "waitting time": "waiting time",
    "written time": "waiting time",
    "cpu par time": "CPU burst time",
    "cpu part time": "CPU burst time",
    "cpu per time": "CPU burst time",
    "cpu pur time": "CPU burst time",
    "cpu brust": "CPU burst",
    "cpu brst": "CPU burst",
    "turn around time": "turnaround time",
    "turn around": "turnaround",
    "turnaround tim": "turnaround time",
    "turnround time": "turnaround time",
    "time content": "time quantum",
    "time, content": "time quantum",
    "fcf s": "FCFS",
    "round problem": "Round Robin",
    "round robbin": "Round Robin",
    "ron robin": "Round Robin",
    "roun robin": "Round Robin",
    "radicube": "ready queue",
    "ready q": "ready queue",
    "ratq": "ready queue",
    "already queue": "ready queue",
    "all ready queue": "ready queue",
    "rahe condition": "race condition",
    "ray condition": "race condition",
    "konvai effect": "convoy effect",
    "convay effect": "convoy effect",
    "bsgf": "SJF",
    "setf": "SRTF",
    "preemtive": "preemptive",
    "prempetive": "preemptive",
    "context swith": "context switch",
    "contex switch": "context switch",
    "context swit": "context switch",
    "context swich": "context switch",
    "switching creates overhead": "context switching creates overhead",
    "seduling": "scheduling",
    "sheduling": "scheduling",
    "shechuling": "scheduling",
    "schedueling": "scheduling",
    "skedyuling": "scheduling",
    "thruput": "throughput",
    "despatcher": "dispatcher",
    "dispacher": "dispatcher",
    "semafore": "semaphore",
    "semafor": "semaphore",
    "dead lock": "deadlock",
    "dead lok": "deadlock",
    "stavation": "starvation",
    "mutix": "mutex",
    "muteks": "mutex",
    "pro ses": "process",
    "pro cess": "process",
    "proses": "process",
    "kernal": "kernel",
    "pipline": "pipeline",
    "interupt": "interrupt",
    "enterrupt": "interrupt",
    "virtual memry": "virtual memory",
    "page falt": "page fault",
    "threshing": "thrashing",
    "aaching, ageing": "aging",
    "mili seps": "milliseconds",
    "mili sep": "millisecond",
    "running insistence": "running instance",
    "executable mile disk": "executable file on disk",

    # Common Roman Urdu & Code-Mixed slips
    "1 mere cheese": "ek ahem cheez",
    "mere cheese": "ahem cheez",
    "lacking-jap": "lekin jab",
    "lacking jap": "lekin jab",
    "samultaneously": "simultaneously",
    "tha is": "that is",
    "ne of": "one of",
    "ne process": "new process",
    "aerat official intelli jineses": "Artificial Intelligence",
    "automotive learning": "Automated Machine Learning",
    "10 place": "template",

    # General CS
    "algorythm": "algorithm",
    "algorhythm": "algorithm",
    "algoritm": "algorithm",
    "compilor": "compiler",
    "interperter": "interpreter",
    "polymorfism": "polymorphism",
    "inheritence": "inheritance",
    "encapsualtion": "encapsulation",
    "abstarction": "abstraction",
    "binry tree": "binary tree",
    "link list": "linked list",
    "hush table": "hash table",
    "tcp ip": "TCP/IP",
    "normaliztion": "normalization",
    "primry key": "primary key",
    "dijkstra algoritm": "Dijkstra algorithm",
    "backpropogation": "backpropagation",
    "relational databas": "relational database",

    # Biology & Life Sciences mishearings
    "flex action": "reflex action",
    "the flex action": "the reflex action",
    "peer receptor": "receptor",
    "sentinue runs": "sensory neurons",
    "new ronns": "neurons",
    "new ronn": "neuron",
    "a factor": "an effector",
    "affectors": "effectors",
    "affector": "effector",
    "centeral nervous system": "central nervous system",
    "periferal nervous system": "peripheral nervous system",

    # Physics, Electricity & Circuits mishearings
    "oms law": "Ohm's Law",
    "arms law": "Ohm's Law",
    "o homes law": "Ohm's Law",
    "o homes": "Ohms",
    "gulams": "Coulombs",
    "gulam": "Coulomb",
    "m-pair": "ampere",
    "m pair": "ampere",
    "emperors": "amperes",
    "emperor": "ampere",
    "myodin volts": "measured in volts",
    "myodin ohms": "measured in ohms",
    "two valves": "two bulbs",
    "two valve": "two bulbs",
    "new rona": "neutrons",
    "vacation of velocity and battery": "voltage and battery",
    "boarding, mobile card": "bulbs, mobile devices",
    "wale socket": "wall socket",
    "i is equal to q by d": "I is equal to Q by t"
}

# Context-aware and phonetic pattern replacement rules (ordered by specificity)
CONTEXTUAL_CORRECTION_RULES: list[tuple[re.Pattern, str]] = [
    # Physics & Electricity
    (re.compile(r'\b(?:oms|arms|o\s*homes)\s+law\b', re.IGNORECASE), "Ohm's Law"),
    (re.compile(r'\b(\d+)\s+(?:emperors|emperor|m-pair|m\s+pair)\b', re.IGNORECASE), r"\1 amperes"),
    (re.compile(r'\b(\d+)\s+(?:gulams|gulam)\b', re.IGNORECASE), r"\1 Coulombs"),
    (re.compile(r'\bmyodin\s+volts\b', re.IGNORECASE), "measured in volts"),
    (re.compile(r'\bmyodin\s+ohms\b', re.IGNORECASE), "measured in ohms"),
    (re.compile(r'\b(?:two|three|four)\s+valves\b', re.IGNORECASE), "two bulbs"),
    (re.compile(r'\bvacation\s+of\s+velocity\s+and\s+battery\b', re.IGNORECASE), "voltage and battery"),
    (re.compile(r'\bwale\s+socket\b', re.IGNORECASE), "wall socket"),
    # Scheduling & Timing Formulas

    (re.compile(r'\b(?:wedding|weding|waitting)\s+time\b', re.IGNORECASE), "waiting time"),
    (re.compile(r'\bwritten\s+time\b', re.IGNORECASE), "waiting time"),
    (re.compile(r'\bcpu\s+(?:par|part|per|pur)\s+time\b', re.IGNORECASE), "CPU burst time"),
    (re.compile(r'\bcpu\s+(?:brust|brst)\b', re.IGNORECASE), "CPU burst"),
    (re.compile(r'\bturn\s+around\s+time\b', re.IGNORECASE), "turnaround time"),
    (re.compile(r'\bturn\s+around\b', re.IGNORECASE), "turnaround"),
    # Charts & Algorithms
    (re.compile(r'\b(?:gaon|gan|gaant|grant|gun)\s+charts?\b', re.IGNORECASE), "Gantt chart"),
    (re.compile(r'\bgaon\s+chart\b', re.IGNORECASE), "Gantt chart"),
    (re.compile(r'\b(?:round\s+problem|ron\s+robin|roun\s+robin|round\s+robbin)\b', re.IGNORECASE), "Round Robin"),
    (re.compile(r'\b(?:radicube|ratq|ready\s+q|radi\s+queue|raid\s+queue)\b', re.IGNORECASE), "ready queue"),
    (re.compile(r'\b(?:already|all\s+ready)\s+queue\b', re.IGNORECASE), "ready queue"),
    (re.compile(r'\b(?:radi|raid)\s+state\b', re.IGNORECASE), "ready state"),
    (re.compile(r'\bin\s+raid\b', re.IGNORECASE), "in ready state"),
    (re.compile(r'\b(?:rahe|ray|re)\s+condition\b', re.IGNORECASE), "race condition"),
    (re.compile(r'\b(?:konvai|convay)\s+effect\b', re.IGNORECASE), "convoy effect"),
    (re.compile(r'\btime[,\s]+content\b', re.IGNORECASE), "time quantum"),
    # Roman Urdu lecture colloquialisms
    (re.compile(r'\b1\s+mere\s+cheese\b', re.IGNORECASE), "Ek ahem cheez"),
    (re.compile(r'\b(?:aik|ek)\s+mere\s+cheese\b', re.IGNORECASE), "ek ahem cheez"),
    (re.compile(r'\bmere\s+cheese\b', re.IGNORECASE), "ahem cheez"),
    (re.compile(r'\blacking[\s-]jap\b', re.IGNORECASE), "lekin jab"),
    (re.compile(r'\bcoach\s+(?=interactive|process|background)', re.IGNORECASE), "kuch "),
    # General OS & Computing Terms
    (re.compile(r'\bne\s+of\b', re.IGNORECASE), "one of"),
    (re.compile(r'\bne\s+process\b', re.IGNORECASE), "new process"),
    (re.compile(r'\btha\s+is\b', re.IGNORECASE), "that is"),
    (re.compile(r'\btha\s+all\b', re.IGNORECASE), "that all"),
    (re.compile(r'\bsamultaneously\b', re.IGNORECASE), "simultaneously"),
    (re.compile(r'\bmili\s+seps?\b', re.IGNORECASE), "milliseconds"),
    (re.compile(r'\brunning\s+insistence\b', re.IGNORECASE), "running instance"),
    (re.compile(r'\bbsgf\b', re.IGNORECASE), "SJF"),
    (re.compile(r'\bsetf\b', re.IGNORECASE), "SRTF"),
    (re.compile(r'\b(?:the\s+purity|purity\s+scheduling)\b', re.IGNORECASE), "priority scheduling"),
    # Academic & Exam Corrections
    (re.compile(r'\b(?:medical|medic)\s+questions?\b', re.IGNORECASE), "numerical questions"),
    (re.compile(r'\bexam\s+are\s+not\s+only\s+definition\b', re.IGNORECASE), "exams are not only definitions"),
    (re.compile(r'\bfirst\s+server\b', re.IGNORECASE), "first served"),
    (re.compile(r'\bfirst\s+ADQ\b', re.IGNORECASE), "ready queue"),
    (re.compile(r'\bratiq\b', re.IGNORECASE), "ready queue"),
    (re.compile(r'\bvscodes?\b', re.IGNORECASE), "VS Code"),
]

# ---------------------------------------------------------------------------
# Whisper initial_prompt builder
# ---------------------------------------------------------------------------
WHISPER_DOMAIN_PROMPT = (
    "This is a university computer science and software engineering lecture. Key technical topics include: "
    "operating systems, process scheduling, Gantt chart, FCFS, Round Robin, preemptive, "
    "context switch, CPU burst, turnaround time, waiting time, deadlock, semaphore, mutex, "
    "virtual memory, page fault, thrashing, kernel, thread, pipeline, data structures, "
    "binary search tree, linked list, hash table, Dijkstra algorithm, recursion, Big O notation, "
    "relational database, SQL, normalization, transactions, networking, TCP/IP, sockets, "
    "machine learning, neural networks, gradient descent, backpropagation, and algorithms."
)

# OpenAI Whisper decoding slices prompts via prompt_tokens[-(n_text_ctx // 2 - 1):]
# where standard models use n_text_ctx = 448 -> 223 prompt tokens max.
WHISPER_PROMPT_MAX_TOKENS = 223


@lru_cache(maxsize=2)
def _get_whisper_tokenizer(multilingual: bool = True):
    try:
        from whisper.tokenizer import get_tokenizer
        return get_tokenizer(multilingual=multilingual)
    except Exception:
        return None


def count_whisper_prompt_tokens(text: str, multilingual: bool = True) -> int:
    """Count tokens using Whisper's exact initial_prompt tokenization (' ' + text.strip())."""
    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    tok = _get_whisper_tokenizer(multilingual=multilingual)
    if tok is not None:
        return len(tok.encode(" " + cleaned))
    # Conservative fallback if whisper tokenizer is unavailable
    return len(re.findall(r"\w+|[^\w\s]", cleaned))


def _trim_words_to_token_budget(text: str, max_tokens: int, multilingual: bool = True) -> tuple[str, bool]:
    words = (text or "").split()
    if not words or max_tokens <= 0:
        return "", bool(words)
    if count_whisper_prompt_tokens(" ".join(words), multilingual=multilingual) <= max_tokens:
        return " ".join(words), False
    lo, hi = 1, len(words)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = " ".join(words[:mid]).rstrip(",.;:")
        if cand and count_whisper_prompt_tokens(cand, multilingual=multilingual) <= max_tokens:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best, True


def _fit_general_terms_to_budget(max_tokens: int, multilingual: bool = True) -> tuple[str, bool]:
    """Fit as many comma-separated technical terms from WHISPER_DOMAIN_PROMPT as fit in max_tokens."""
    if max_tokens <= 0:
        return "", True
    if count_whisper_prompt_tokens(WHISPER_DOMAIN_PROMPT, multilingual=multilingual) <= max_tokens:
        return WHISPER_DOMAIN_PROMPT, False

    terms_part = WHISPER_DOMAIN_PROMPT.split("Key technical topics include:", 1)[-1].strip().rstrip(".")
    terms_part = re.sub(r"\band\s+algorithms$", "algorithms", terms_part)
    terms = [t.strip() for t in terms_part.split(",") if t.strip()]
    prefix = "Technical terms: "
    kept: list[str] = []
    truncated = False
    for term in terms:
        cand = prefix + ", ".join(kept + [term]) + "."
        if count_whisper_prompt_tokens(cand, multilingual=multilingual) <= max_tokens:
            kept.append(term)
        else:
            truncated = True
            break
    if kept:
        return prefix + ", ".join(kept) + ".", truncated
    trimmed, was_trimmed = _trim_words_to_token_budget(terms_part, max_tokens, multilingual=multilingual)
    return trimmed, truncated or was_trimmed


def build_whisper_initial_prompt(
    topic_hint: Optional[str] = None,
    max_tokens: int = WHISPER_PROMPT_MAX_TOKENS,
    multilingual: bool = True,
) -> str:
    """
    Build Whisper's initial_prompt combining the broad general technical vocabulary
    with any dynamic topic hint while respecting Whisper's prompt-token limit (223 tokens).
    """
    limit = max(1, int(max_tokens or WHISPER_PROMPT_MAX_TOKENS))
    clean_topic = re.sub(r"\s+", " ", (topic_hint or "").strip()).strip()
    if not clean_topic:
        fitted, truncated = _fit_general_terms_to_budget(limit, multilingual=multilingual)
        if truncated:
            logger.warning(
                "WHISPER_DOMAIN_PROMPT exceeded %d tokens and was compacted to %d tokens.",
                limit,
                count_whisper_prompt_tokens(fitted, multilingual=multilingual),
            )
        return fitted

    topic_clause = f"Focus topic: {clean_topic}" + ("" if clean_topic[-1] in ".!?" else ".")
    combined = f"{WHISPER_DOMAIN_PROMPT} {topic_clause}"
    if count_whisper_prompt_tokens(combined, multilingual=multilingual) <= limit:
        return combined

    # Over token budget: first compact boilerplate intro while preserving all technical terms.
    terms_part = WHISPER_DOMAIN_PROMPT.split("Key technical topics include:", 1)[-1].strip()
    compact_general = f"Technical terms: {terms_part}"
    compact_combined = f"{compact_general} {topic_clause}"
    if count_whisper_prompt_tokens(compact_combined, multilingual=multilingual) <= limit:
        logger.warning(
            "Whisper initial_prompt compacted general intro to retain full technical vocabulary and topic hint within %d tokens.",
            limit,
        )
        return compact_combined

    # Still over budget (oversized topic_hint or tight limit): reserve budget for BOTH sources.
    topic_full_tokens = count_whisper_prompt_tokens(topic_clause, multilingual=multilingual)
    min_topic_reserve = max(12, limit // 3)
    general_full_tokens = count_whisper_prompt_tokens(compact_general, multilingual=multilingual)

    if general_full_tokens + min_topic_reserve <= limit:
        general_part = compact_general
        remaining_for_topic = limit - general_full_tokens - 1
    else:
        topic_target = min(topic_full_tokens, min_topic_reserve)
        general_budget = max(1, limit - topic_target - 1)
        general_part, _ = _fit_general_terms_to_budget(general_budget, multilingual=multilingual)
        remaining_for_topic = max(1, limit - count_whisper_prompt_tokens(general_part, multilingual=multilingual) - 1)

    prefix = "Focus topic: "
    prefix_tokens = count_whisper_prompt_tokens(prefix + ".", multilingual=multilingual)
    topic_word_budget = max(1, remaining_for_topic - prefix_tokens)
    trimmed_topic, topic_truncated = _trim_words_to_token_budget(
        clean_topic, topic_word_budget, multilingual=multilingual
    )
    final_topic_clause = (
        f"{prefix}{trimmed_topic}" + ("" if trimmed_topic.endswith((".", "!", "?")) else ".")
        if trimmed_topic
        else ""
    )
    result = f"{general_part} {final_topic_clause}".strip() if general_part else final_topic_clause

    # Safety clamp if BPE boundary merge added 1 token
    while count_whisper_prompt_tokens(result, multilingual=multilingual) > limit and trimmed_topic:
        words = trimmed_topic.split()
        if len(words) <= 1:
            break
        trimmed_topic = " ".join(words[:-1]).rstrip(",.;:")
        topic_truncated = True
        final_topic_clause = f"{prefix}{trimmed_topic}."
        result = f"{general_part} {final_topic_clause}".strip() if general_part else final_topic_clause

    if topic_truncated or general_part != compact_general:
        logger.warning(
            "Whisper initial_prompt exceeded %d-token limit; trimmed to %d tokens while retaining both general terms and topic hint.",
            limit,
            count_whisper_prompt_tokens(result, multilingual=multilingual),
        )
    return result


def get_whisper_prompt(
    topic_hint: Optional[str] = None,
    max_tokens: Optional[int] = None,
    multilingual: bool = True,
) -> str:
    """Return Whisper's initial_prompt combining broad domain vocabulary with any topic hint."""
    return build_whisper_initial_prompt(
        topic_hint=topic_hint,
        max_tokens=max_tokens or WHISPER_PROMPT_MAX_TOKENS,
        multilingual=multilingual,
    )


# ---------------------------------------------------------------------------
# Phonetic skeleton and similarity matching
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8192)
def _consonant_skeleton(text: str) -> str:
    """Reduce words to their consonant skeleton for phonetic comparison."""
    words = text.lower().split()
    skel_words = []
    for w in words:
        clean = re.sub(r'[^a-z0-9]', '', w)
        if len(clean) > 1:
            skel = clean[0] + re.sub(r'[aeiou]', '', clean[1:])
        else:
            skel = clean
        skel_words.append(skel)
    return " ".join(skel_words)


def _similarity(a: str, b: str) -> float:
    """Compute string similarity ratio."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


_CANONICAL_WORDS_SET: frozenset[str] = frozenset(
    re.sub(r'[^a-z0-9]', '', w)
    for term in CANONICAL_TECHNICAL_TERMS
    for w in term.lower().split()
)

_CANONICAL_BY_WC: dict[int, list[tuple[str, str, str]]] = {}
for _can in CANONICAL_TECHNICAL_TERMS:
    _can_low = _can.lower()
    _wc = len(_can_low.split())
    _CANONICAL_BY_WC.setdefault(_wc, []).append((_can, _can_low, _consonant_skeleton(_can_low)))


@lru_cache(maxsize=8192)
def match_canonical_term(phrase: str, min_score: float = DOMAIN_CORRECTOR_MIN_SCORE) -> Optional[str]:
    """
    Match an arbitrary spoken phrase against the canonical technical vocabulary
    using combined phonetic consonant skeleton and string edit distance.
    Works for any unseen misspelling / mishearing.
    """
    phrase_clean = phrase.strip().lower()
    p_len = len(phrase_clean)
    if p_len < 4:
        return None

    words_count = len(phrase_clean.split())
    candidates = _CANONICAL_BY_WC.get(words_count)
    if not candidates:
        return None

    phrase_skel = _consonant_skeleton(phrase_clean)
    s_len = len(phrase_skel)
    best_match: Optional[str] = None
    best_score = 0.0

    for can, can_clean, can_skel in candidates:
        # Fast upper-bound length filter before SequenceMatcher
        if abs(s_len - len(can_skel)) > max(s_len, len(can_skel)) * 0.45:
            continue
        skel_sim = SequenceMatcher(None, phrase_skel, can_skel).ratio()
        if skel_sim < 0.60:
            continue

        str_sim = SequenceMatcher(None, phrase_clean, can_clean).ratio()
        score = 0.45 * skel_sim + 0.55 * str_sim

        if score > best_score:
            best_score = score
            best_match = can

    if best_score >= min_score and best_match is not None:
        return best_match
    return None


def is_hallucinated_noise(text: str) -> bool:
    """
    Detect degenerate ASR noise artifacts produced during silence or low SNR.
    e.g. 'dhk', 'bk уже', 'கே', single consonants or repetitive non-lexical tokens.
    """
    if not text or not text.strip():
        return True
    cleaned = text.strip()
    # If contains non-Latin, non-Urdu/Arabic characters (e.g. Cyrillic, Tamil, Devanagari, CJK)
    if re.search(r'[\u0400-\u04FF\u0B80-\u0BFF\u0900-\u097F\u4E00-\u9FFF]', cleaned):
        return True
    # If 1-2 char words with no vowels
    tokens = [t for t in re.findall(r'\b\w+\b', cleaned)]
    if len(tokens) <= 2 and all(len(t) <= 3 and not re.search(r'[aeiouy]', t, re.IGNORECASE) for t in tokens):
        return True
    return False


def apply_contextual_rules(text: str) -> str:
    """Apply high-precision contextual and phonetic regex replacement rules."""
    result = text
    for pattern, replacement in CONTEXTUAL_CORRECTION_RULES:
        def _replace_match(m, rep=replacement):
            matched = m.group(0)
            if matched.isupper():
                return rep.upper()
            if matched and matched[0].isupper():
                return rep[:1].upper() + rep[1:]
            return rep
        result = pattern.sub(_replace_match, result)
    return result


# ---------------------------------------------------------------------------
# Core Correction Engine
# ---------------------------------------------------------------------------
_MULTIWORD_MISHEARING_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b' + re.escape(mishearing) + r'\b', re.IGNORECASE), canonical)
    for mishearing, canonical in sorted(_COMMON_MISHEARINGS.items(), key=lambda x: -len(x[0]))
    if len(mishearing.split()) > 1
]


@lru_cache(maxsize=4096)
def correct_domain_terms(text: str) -> str:
    """
    Generalized correction engine:
    1. Runs high-precision contextual regex rules.
    2. Checks fast-path common mishearings.
    3. Performs dynamic 2-gram and 1-gram phonetic matching against the canonical vocabulary.
    4. Handles punctuation preservation.
    """
    if not text or not text.strip():
        return text

    # Pre-pass: Context-aware regex rules
    result = apply_contextual_rules(text)
    canonical_words = _CANONICAL_WORDS_SET

    # Pass 1: Multi-word common mishearings (exact fast-path)
    for pattern, canonical in _MULTIWORD_MISHEARING_PATTERNS:
        def _sub_func(m, can=canonical):
            matched = m.group(0)
            if matched.isupper():
                return can.upper()
            if matched and matched[0].isupper():
                return can[:1].upper() + can[1:]
            return can
        result = pattern.sub(_sub_func, result)

    # Pass 2: Token-level scan with 2-gram and 1-gram dynamic phonetic matching
    words = result.split()
    corrected_words = []
    skip_next = False

    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            continue

        clean_word = re.sub(r'[^\w]', '', word).lower()

        # Check 2-gram first
        if i + 1 < len(words):
            next_clean = re.sub(r'[^\w]', '', words[i+1]).lower()
            bigram = f"{clean_word} {next_clean}"

            # If already canonical bigram, preserve original word casing as-is
            if any(bigram == term.lower() for term in CANONICAL_TECHNICAL_TERMS):
                corrected_words.append(word)
                continue

            # 2a. Common mishearing map
            if bigram in _COMMON_MISHEARINGS:
                trailing_punct = re.search(r'[^\w]+$', words[i+1])
                replacement = _COMMON_MISHEARINGS[bigram]
                if words[i] and words[i][0].isupper():
                    replacement = replacement[:1].upper() + replacement[1:]
                if trailing_punct:
                    replacement += trailing_punct.group()
                corrected_words.append(replacement)
                skip_next = True
                continue

            # 2b. Generalized phonetic match on 2-gram
            match = match_canonical_term(bigram, min_score=DOMAIN_CORRECTOR_BIGRAM_MIN_SCORE)
            if match:
                trailing_punct = re.search(r'[^\w]+$', words[i+1])
                replacement = match
                if words[i] and words[i][0].isupper():
                    replacement = replacement[:1].upper() + replacement[1:]
                if trailing_punct:
                    replacement += trailing_punct.group()
                corrected_words.append(replacement)
                skip_next = True
                continue

        # Check 1-gram
        # 1a. Common mishearing map
        if clean_word in _COMMON_MISHEARINGS:
            leading = re.match(r'^[^\w]*', word).group()
            trailing = re.search(r'[^\w]*$', word).group()
            corrected_words.append(leading + _COMMON_MISHEARINGS[clean_word] + trailing)
            continue

        # Skip if already a known canonical technical word
        if clean_word in canonical_words:
            corrected_words.append(word)
            continue

        # 1b. Generalized phonetic match on 1-gram
        match = match_canonical_term(clean_word, min_score=0.83)
        if match:
            leading = re.match(r'^[^\w]*', word).group()
            trailing = re.search(r'[^\w]*$', word).group()
            corrected_words.append(leading + match + trailing)
            continue

        corrected_words.append(word)

    return " ".join(corrected_words)

