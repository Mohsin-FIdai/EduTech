import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
"""
Configuration settings and constraints for the Lecture & Meeting Intelligence System.
"""

from pathlib import Path
import os

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
PROCESSED_DIR = STORAGE_DIR / "processed"
CHUNKS_DIR = STORAGE_DIR / "chunks"

# Ensure directories exist
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Chunking Parameters (Phase 3)
TARGET_CHUNK_DURATION_SECONDS = 25.0
MAX_CHUNK_DURATION_SECONDS = 30.0
MIN_CHUNK_DURATION_SECONDS = 2.0
DEFAULT_CHUNK_OVERLAP_SECONDS = 0.5
SILENCE_THRESHOLD_DBFS = -30.0
MIN_PAUSE_DURATION_SECONDS = 0.20

# Compute Device Auto-Detection (CUDA GPU with CPU fallback)
def _detect_compute_device() -> str:
    override = os.getenv("COMPUTE_DEVICE", "").strip().lower()
    if override in ("cuda", "cpu"):
        return override
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

DEFAULT_COMPUTE_DEVICE = _detect_compute_device()
AUDIO_DEVICE = os.getenv("AUDIO_DEVICE", DEFAULT_COMPUTE_DEVICE)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", DEFAULT_COMPUTE_DEVICE)

# Transcription Parameters (Phase 4)
DEFAULT_WHISPER_MODEL = os.getenv("DEFAULT_WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", DEFAULT_COMPUTE_DEVICE)
SUPPORTED_WHISPER_MODELS = (
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3", "large-v3-turbo", "turbo"
)
WHISPER_NO_SPEECH_THRESHOLD = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.6"))
WHISPER_COMPRESSION_RATIO_THRESHOLD = float(os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.2"))
WHISPER_LOGPROB_THRESHOLD = float(os.getenv("WHISPER_LOGPROB_THRESHOLD", "-1.0"))
WHISPER_TEMPERATURE_FALLBACK = (0.0, 0.2)
WHISPER_CONDITION_ON_PREVIOUS_TEXT = False

from roman_urdu_nlp.utils import env_flag

# LLM Configuration (Local & Cloud Provider Abstraction)
import sys
ENABLE_LOCAL_LLM = (
    env_flag("ENABLE_LOCAL_LLM", default=True)
    and (os.getenv("TESTING") != "1")
    and ("pytest" not in sys.modules)
)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
NOTES_LLM_PROVIDER = os.getenv("NOTES_LLM_PROVIDER", "ollama")
SUMMARY_LLM_PROVIDER = os.getenv("SUMMARY_LLM_PROVIDER", "ollama")
TRANSLATION_LLM_PROVIDER = os.getenv("TRANSLATION_LLM_PROVIDER", "ollama")
QUIZ_LLM_PROVIDER = os.getenv("QUIZ_LLM_PROVIDER", "ollama")
RAG_LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "ollama")
ALLOW_OLLAMA_TO_LOCAL_FALLBACK = env_flag("ALLOW_OLLAMA_TO_LOCAL_FALLBACK", default=False)
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LOCAL_LLM_DEVICE = os.getenv("LOCAL_LLM_DEVICE", DEFAULT_COMPUTE_DEVICE)
LOCAL_LLM_MAX_TOKENS = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "512"))
LOCAL_LLM_TEMPERATURE = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.1"))

# Optional Ollama Provider Configuration (used when LLM_PROVIDER=ollama)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "300"))
OLLAMA_THINK = env_flag("OLLAMA_THINK", default=False)

# ─── LLM Repetition-Loop Protection ────────────────────────────────
# N-gram window size for repetition detection (e.g. 4-gram)
LLM_REPETITION_NGRAM_SIZE = int(os.getenv("LLM_REPETITION_NGRAM_SIZE", "4"))
# Maximum times a single n-gram may appear before the output is flagged
LLM_REPETITION_MAX_REPEATS = int(os.getenv("LLM_REPETITION_MAX_REPEATS", "3"))
# Fraction of lines that must be duplicates before line-level loop is flagged
LLM_REPETITION_LINE_THRESHOLD = float(os.getenv("LLM_REPETITION_LINE_THRESHOLD", "0.4"))

# ─── RAG / Grounding ───────────────────────────────────────────────
RAG_MIN_TOKEN_OVERLAP = float(os.getenv("RAG_MIN_TOKEN_OVERLAP", "0.80"))
MAX_CONVERSATION_TURNS = int(os.getenv("MAX_CONVERSATION_TURNS", "5"))
RAG_SLIDING_WINDOW_THRESHOLD = float(os.getenv("RAG_SLIDING_WINDOW_THRESHOLD", "3.5"))
RAG_TOPIC_MATCH_MIN_SCORE = float(os.getenv("RAG_TOPIC_MATCH_MIN_SCORE", "4.0"))

# ─── Pedagogical Validation ────────────────────────────────────────
NOTES_GROUNDING_FLOOR = float(os.getenv("NOTES_GROUNDING_FLOOR", "0.25"))
NOTES_GROUNDING_NORMALIZATION = float(os.getenv("NOTES_GROUNDING_NORMALIZATION", "0.50"))

# ─── Contamination / Grounding Isolation ──────────────────────────────────────
# When True: static reference dicts (_STANDARD_DEFINITIONS, DOMAIN_KNOWLEDGE_BASES)
# are fully disabled. Every sentence in notes, quizzes and answers must derive
# from the current recording's transcript only.
STRICT_TRANSCRIPT_ONLY = env_flag("STRICT_TRANSCRIPT_ONLY", default=False)

# ─── Academic Intelligence v2 Upgrade ─────────────────────────────────────────
ACADEMIC_INTEL_V2 = env_flag("ACADEMIC_INTEL_V2", default=False)
EXAM_CUE_MIN_CONFIDENCE = float(os.getenv("EXAM_CUE_MIN_CONFIDENCE", "0.65"))
ALIGN_COS_MIN = float(os.getenv("ALIGN_COS_MIN", "0.45"))
ALIGN_OVERLAP_MIN = float(os.getenv("ALIGN_OVERLAP_MIN", "0.20"))
MCQ_DEDUPE_COS = float(os.getenv("MCQ_DEDUPE_COS", "0.85"))
MIN_WORDS_FOR_10 = int(os.getenv("MIN_WORDS_FOR_10", "600"))
MAX_REPAIR_PER_ITEM = int(os.getenv("MAX_REPAIR_PER_ITEM", "2"))
INTEL_TIME_BUDGET_S = float(os.getenv("INTEL_TIME_BUDGET_S", "45.0"))

# ─── Context Stitching ─────────────────────────────────────────────
STITCH_MAX_OVERLAP_WORDS = int(os.getenv("STITCH_MAX_OVERLAP_WORDS", "4"))
STITCH_TIME_GAP_THRESHOLD = float(os.getenv("STITCH_TIME_GAP_THRESHOLD", "3.0"))
STITCH_CLOSE_GAP_THRESHOLD = float(os.getenv("STITCH_CLOSE_GAP_THRESHOLD", "1.5"))
CHUNK_OVERLAP_FUZZY_THRESHOLD = float(os.getenv("CHUNK_OVERLAP_FUZZY_THRESHOLD", "0.85"))
CHUNK_OVERLAP_MAX_WORDS = int(os.getenv("CHUNK_OVERLAP_MAX_WORDS", "8"))

# ─── Embedding / Semantic Search ───────────────────────────────────
SEMANTIC_SEARCH_THRESHOLD = float(os.getenv("SEMANTIC_SEARCH_THRESHOLD", "0.40"))
SEMANTIC_QA_THRESHOLD = float(os.getenv("SEMANTIC_QA_THRESHOLD", "0.50"))

# ─── Translation ───────────────────────────────────────────────────
ENABLE_ONLINE_TRANSLATION = env_flag("ENABLE_ONLINE_TRANSLATION", default=True)
ENGLISH_PASSTHROUGH_RATIO = float(os.getenv("ENGLISH_PASSTHROUGH_RATIO", "0.80"))
TRANSLATION_CONFIDENCE_GOOGLE = float(os.getenv("TRANSLATION_CONFIDENCE_GOOGLE", "0.95"))
TRANSLATION_CONFIDENCE_LLM = float(os.getenv("TRANSLATION_CONFIDENCE_LLM", "0.85"))
TRANSLATION_CONFIDENCE_DICTIONARY = float(os.getenv("TRANSLATION_CONFIDENCE_DICTIONARY", "0.60"))

# ─── Audio Preprocessing ───────────────────────────────────────────
AUDIO_TARGET_PEAK_NORM = float(os.getenv("AUDIO_TARGET_PEAK_NORM", "0.95"))
AUDIO_DENOISE_PROP_DECREASE = float(os.getenv("AUDIO_DENOISE_PROP_DECREASE", "0.75"))
VAD_TOP_DB = float(os.getenv("VAD_TOP_DB", "20.0"))
VAD_MIN_SPEECH_DURATION_SEC = float(os.getenv("VAD_MIN_SPEECH_DURATION_SEC", "0.1"))

# ─── Domain Corrector ──────────────────────────────────────────────
DOMAIN_CORRECTOR_MIN_SCORE = float(os.getenv("DOMAIN_CORRECTOR_MIN_SCORE", "0.82"))
DOMAIN_CORRECTOR_BIGRAM_MIN_SCORE = float(os.getenv("DOMAIN_CORRECTOR_BIGRAM_MIN_SCORE", "0.84"))

# ─── Contextual Transcription ──────────────────────────────────────
HALLUCINATION_TOKEN_RATIO = float(os.getenv("HALLUCINATION_TOKEN_RATIO", "0.35"))

# Database Configuration (SQLite with WAL mode)
# If running under pytest or TESTING environment variable, automatically isolate to test DB
import sys
if os.getenv("TESTING") == "1" or "pytest" in sys.modules:
    DEFAULT_DB = STORAGE_DIR / "test_isolated.db"
else:
    DEFAULT_DB = BASE_DIR / "lecture_system.db"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB}")

# Section 8 Fixed Constraints
MAX_AUDIO_SIZE_BYTES = 50 * 1024 * 1024       # 50 MB
MAX_VIDEO_SIZE_BYTES = 400 * 1024 * 1024      # 400 MB
MAX_DURATION_SECONDS = 30 * 60                # 30 minutes (1800 seconds)

# Supported Extensions
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov"}
ALL_SUPPORTED_EXTENSIONS = SUPPORTED_AUDIO_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS

# MIME types mapping for extra validation
ALLOWED_MIME_PREFIXES = ("audio/", "video/")
