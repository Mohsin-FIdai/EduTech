# Thresholds & Configuration Constants

All constants are defined in [`app/config.py`](../app/config.py) and overridable via environment variables.

## Base Directories & Storage

| Constant | Default | Env Var | Used In |
|---|---|---|---|
| `BASE_DIR` | Project root directory | – | `config.py` |
| `STORAGE_DIR` | `BASE_DIR / "storage"` | – | `config.py` |
| `UPLOADS_DIR` | `STORAGE_DIR / "uploads"` | – | `storage.py` |
| `PROCESSED_DIR` | `STORAGE_DIR / "processed"` | – | `audio_preprocessor.py`, `storage.py` |
| `CHUNKS_DIR` | `STORAGE_DIR / "chunks"` | – | `audio_chunker.py` |
| `DEFAULT_DB` | `storage/test_isolated.db` (test) or `lecture_system.db` | – | `config.py`, `database.py` |
| `DATABASE_URL` | `sqlite:///{DEFAULT_DB}` | `DATABASE_URL` | `database.py` |

## Audio Chunking

| Constant | Default | Env Var | Used In |
|---|---|---|---|
| `TARGET_CHUNK_DURATION_SECONDS` | `25.0` s | – | `audio_chunker.py` |
| `MAX_CHUNK_DURATION_SECONDS` | `30.0` s | – | `audio_chunker.py` |
| `MIN_CHUNK_DURATION_SECONDS` | `2.0` s | – | `audio_chunker.py` |
| `DEFAULT_CHUNK_OVERLAP_SECONDS` | `0.5` s | – | `audio_chunker.py` |
| `SILENCE_THRESHOLD_DBFS` | `-30.0` dBFS | – | `audio_chunker.py` |
| `MIN_PAUSE_DURATION_SECONDS` | `0.20` s | – | `audio_chunker.py` |

## Whisper / Transcription

| Constant | Default | Env Var | Used In |
|---|---|---|---|
| `DEFAULT_WHISPER_MODEL` | `"base"` | – | `transcription_service.py` |
| `WHISPER_DEVICE` | `"cpu"` | – | `transcription_service.py` |
| `SUPPORTED_WHISPER_MODELS` | Tuple of 13 supported models | – | `transcription_service.py`, `models.py` |
| `WHISPER_NO_SPEECH_THRESHOLD` | `0.6` | `WHISPER_NO_SPEECH_THRESHOLD` | `transcription_service.py` |
| `WHISPER_COMPRESSION_RATIO_THRESHOLD` | `2.2` | `WHISPER_COMPRESSION_RATIO_THRESHOLD` | `transcription_service.py` |
| `WHISPER_LOGPROB_THRESHOLD` | `-1.0` | `WHISPER_LOGPROB_THRESHOLD` | `transcription_service.py` |
| `WHISPER_TEMPERATURE_FALLBACK` | `(0.0, 0.2)` | – | `transcription_service.py` |
| `WHISPER_CONDITION_ON_PREVIOUS_TEXT` | `False` | – | `transcription_service.py` |
| `DEFAULT_COMPUTE_DEVICE` | `"cuda"` (or `"cpu"`) | `COMPUTE_DEVICE` | `config.py` |
| `AUDIO_DEVICE` | `DEFAULT_COMPUTE_DEVICE` | `AUDIO_DEVICE` | `audio_preprocessor.py`, `audio_chunker.py` |
| `EMBEDDING_DEVICE` | `DEFAULT_COMPUTE_DEVICE` | `EMBEDDING_DEVICE` | `embedding_service.py` |

## LLM Configuration

| Constant | Default | Env Var | Used In |
|---|---|---|---|
| `ENABLE_LOCAL_LLM` | `true` (off in pytest) | `ENABLE_LOCAL_LLM` | `rag_service.py`, `intelligence_service.py` |
| `LLM_PROVIDER` | `"local"` | `LLM_PROVIDER` | `llm_provider.py` |
| `LOCAL_LLM_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | `LOCAL_LLM_MODEL` | `llm_provider.py` |
| `LOCAL_LLM_DEVICE` | `"cpu"` | `LOCAL_LLM_DEVICE` | `llm_provider.py` |
| `LOCAL_LLM_MAX_TOKENS` | `512` | `LOCAL_LLM_MAX_TOKENS` | `llm_provider.py` |
| `LOCAL_LLM_TEMPERATURE` | `0.1` | `LOCAL_LLM_TEMPERATURE` | `llm_provider.py` |

## LLM Repetition-Loop Protection

| Constant | Default | Env Var | Used In |
|---|---|---|---|
| `LLM_REPETITION_NGRAM_SIZE` | `4` | `LLM_REPETITION_NGRAM_SIZE` | `llm_provider.py` — `_detect_repetition` |
| `LLM_REPETITION_MAX_REPEATS` | `3` | `LLM_REPETITION_MAX_REPEATS` | `llm_provider.py` — `_detect_repetition` |
| `LLM_REPETITION_LINE_THRESHOLD` | `0.4` | `LLM_REPETITION_LINE_THRESHOLD` | `llm_provider.py` — `_detect_repetition` |

**Semantics:**
- `LLM_REPETITION_NGRAM_SIZE`: window of consecutive words (tokens after split) that forms one n-gram unit.
- `LLM_REPETITION_MAX_REPEATS`: maximum allowed occurrences of the same n-gram before the output is truncated at that occurrence.
- `LLM_REPETITION_LINE_THRESHOLD`: if the fraction of non-empty lines that are exact duplicates of a previously-seen line exceeds this value, the output is truncated at the first duplicate line.

## RAG / Grounding

| Constant | Default | Env Var | Used In |
|---|---|---|---|
| `RAG_MIN_TOKEN_OVERLAP` | `0.80` | `RAG_MIN_TOKEN_OVERLAP` | `pedagogical_validator.py`, `rag_service.py` |
| `MAX_CONVERSATION_TURNS` | `5` | `MAX_CONVERSATION_TURNS` | `rag_service.py` |
| `RAG_SLIDING_WINDOW_THRESHOLD` | `3.5` | `RAG_SLIDING_WINDOW_THRESHOLD` | `rag_service.py` |
| `RAG_TOPIC_MATCH_MIN_SCORE` | `4.0` | `RAG_TOPIC_MATCH_MIN_SCORE` | `rag_service.py` |

## Pedagogical Validation

| Constant | Default | Env Var | Used In |
|---|---|---|---|
| `NOTES_GROUNDING_FLOOR` | `0.25` | `NOTES_GROUNDING_FLOOR` | `pedagogical_validator.py` |
| `NOTES_GROUNDING_NORMALIZATION` | `0.50` | `NOTES_GROUNDING_NORMALIZATION` | `pedagogical_validator.py` |

## Contamination / Grounding Isolation

| Constant | Default | Env Var | Used In |
|---|---|---|---|
| `STRICT_TRANSCRIPT_ONLY` | `false` | `STRICT_TRANSCRIPT_ONLY` | `rag_service.py`, `study_guide_generator.py`, `lecture_context.py` |

## Academic Intelligence v2 Upgrade

| Constant | Default | Env Var | Used In |
|---|---|---|---|
| `ACADEMIC_INTEL_V2` | `false` | `ACADEMIC_INTEL_V2` | `intelligence_service.py`, `topic_intelligence.py` |
| `EXAM_CUE_MIN_CONFIDENCE` | `0.65` | `EXAM_CUE_MIN_CONFIDENCE` | `discourse_cues.py` |
| `ALIGN_COS_MIN` | `0.45` | `ALIGN_COS_MIN` | `study_guide_generator.py` |
| `ALIGN_OVERLAP_MIN` | `0.20` | `ALIGN_OVERLAP_MIN` | `study_guide_generator.py` |
| `MCQ_DEDUPE_COS` | `0.85` | `MCQ_DEDUPE_COS` | `study_guide_generator.py` |
| `MIN_WORDS_FOR_10` | `600` | `MIN_WORDS_FOR_10` | `study_guide_generator.py` |
| `MAX_REPAIR_PER_ITEM` | `2` | `MAX_REPAIR_PER_ITEM` | `academic_pipeline.py`, `intelligence_service.py` |
| `INTEL_TIME_BUDGET_S` | `45.0` s | `INTEL_TIME_BUDGET_S` | `academic_pipeline.py`, `intelligence_service.py` |

**Academic Intelligence v2 Measured Baselines (CPU-only, Windows Python 3.13):**
- **Discourse Cue Classifier**: `1.00` precision / `1.00` recall on labelled English, Roman Urdu, negated (`nahi aayega`), hedged (`shayad`, `might`), and technical-suppression (`unit test`, `question marks`) benchmark cases (`tests/test_academic_preprocess_and_cues.py`).
- **C-Value Keyphrase Extraction**: Compound technical phrases (`round robin scheduling`, `time quantum`, `context switch`, `turnaround time`, `waiting time`) rank above nested single-word sub-terms with `0` cross-segment boundary bleed (`tests/test_academic_intelligence_core.py`).
- **Grounding & Contamination**: `100%` sentence-to-segment alignment rate ($\ge 95\%$ target) and `0` cross-domain contamination tokens on multi-topic lectures (`tests/test_academic_intelligence.py`).
- **Runtime & Memory Budget**: Full v2 pipeline (preprocessing, C-value keyphrases, cue classification, alignment, 10-MCQ generation, validation, and repair) completes in `~2.1 s` (`< 45.0 s` budget) with `< 15 MB` peak traced heap allocation (`< 200 MB` budget).

## Context Stitching

| Constant | Default | Env Var | Used In |
|---|---|---|---|
| `STITCH_MAX_OVERLAP_WORDS` | `4` | `STITCH_MAX_OVERLAP_WORDS` | `context_stitcher.py` |
| `STITCH_TIME_GAP_THRESHOLD` | `3.0` s | `STITCH_TIME_GAP_THRESHOLD` | `context_stitcher.py` |
| `STITCH_CLOSE_GAP_THRESHOLD` | `1.5` s | `STITCH_CLOSE_GAP_THRESHOLD` | `context_stitcher.py` |
| `CHUNK_OVERLAP_FUZZY_THRESHOLD` | `0.85` | `CHUNK_OVERLAP_FUZZY_THRESHOLD` | `context_stitcher.py` |
| `CHUNK_OVERLAP_MAX_WORDS` | `8` | `CHUNK_OVERLAP_MAX_WORDS` | `context_stitcher.py` |

## Embedding / Semantic Search

| Constant | Default | Env Var | Used In |
|---|---|---|---|
| `SEMANTIC_SEARCH_THRESHOLD` | `0.40` | `SEMANTIC_SEARCH_THRESHOLD` | `embedding_service.py` |
| `SEMANTIC_QA_THRESHOLD` | `0.50` | `SEMANTIC_QA_THRESHOLD` | `embedding_service.py` |

## Translation

| Constant | Default | Env Var | Used In |
|---|---|---|---|
| `ENABLE_ONLINE_TRANSLATION` | `true` | `ENABLE_ONLINE_TRANSLATION` | `text_processor.py`, `translator.py` |
| `ENGLISH_PASSTHROUGH_RATIO` | `0.80` | `ENGLISH_PASSTHROUGH_RATIO` | `text_processor.py` |
| `TRANSLATION_CONFIDENCE_GOOGLE` | `0.95` | `TRANSLATION_CONFIDENCE_GOOGLE` | `text_processor.py` |
| `TRANSLATION_CONFIDENCE_LLM` | `0.85` | `TRANSLATION_CONFIDENCE_LLM` | `text_processor.py` |
| `TRANSLATION_CONFIDENCE_DICTIONARY` | `0.60` | `TRANSLATION_CONFIDENCE_DICTIONARY` | `text_processor.py` |

## Audio Preprocessing

| Constant | Default | Env Var | Used In |
|---|---|---|---|
| `AUDIO_TARGET_PEAK_NORM` | `0.95` | `AUDIO_TARGET_PEAK_NORM` | `audio_preprocessor.py` |
| `AUDIO_DENOISE_PROP_DECREASE` | `0.75` | `AUDIO_DENOISE_PROP_DECREASE` | `audio_preprocessor.py` |
| `VAD_TOP_DB` | `20.0` dB | `VAD_TOP_DB` | `audio_preprocessor.py` |
| `VAD_MIN_SPEECH_DURATION_SEC` | `0.1` s | `VAD_MIN_SPEECH_DURATION_SEC` | `audio_preprocessor.py` |

## Domain Corrector

| Constant | Default | Env Var | Used In |
|---|---|---|---|
| `DOMAIN_CORRECTOR_MIN_SCORE` | `0.82` | `DOMAIN_CORRECTOR_MIN_SCORE` | `domain_corrector.py` |
| `DOMAIN_CORRECTOR_BIGRAM_MIN_SCORE` | `0.84` | `DOMAIN_CORRECTOR_BIGRAM_MIN_SCORE` | `domain_corrector.py` |

## Contextual Transcription

| Constant | Default | Env Var | Used In |
|---|---|---|---|
| `HALLUCINATION_TOKEN_RATIO` | `0.35` | `HALLUCINATION_TOKEN_RATIO` | `contextual_transcription.py` |

## File Size, Duration & Format Constraints

| Constant | Default | Env Var | Used In |
|---|---|---|---|
| `MAX_AUDIO_SIZE_BYTES` | `52,428,800` (50 MB) | – | `audio_validator.py` |
| `MAX_VIDEO_SIZE_BYTES` | `419,430,400` (400 MB) | – | `audio_validator.py` |
| `MAX_DURATION_SECONDS` | `1800` (30 min) | – | `audio_validator.py` |
| `SUPPORTED_AUDIO_EXTENSIONS` | `{.mp3, .wav, .m4a, .ogg, .flac}` | – | `audio_validator.py` |
| `SUPPORTED_VIDEO_EXTENSIONS` | `{.mp4, .webm, .mov}` | – | `audio_validator.py` |
| `ALL_SUPPORTED_EXTENSIONS` | `SUPPORTED_AUDIO_EXTENSIONS \| SUPPORTED_VIDEO_EXTENSIONS` | – | `audio_validator.py` |
| `ALLOWED_MIME_PREFIXES` | `("audio/", "video/")` | – | `audio_validator.py` |

## Feature-Specific LLM Routing & Ollama Lifecycle

| Constant | Default | Env Var | Used In |
|---|---|---|---|
| `NOTES_LLM_PROVIDER` | `"ollama"` | `NOTES_LLM_PROVIDER` | `llm_provider.py` |
| `SUMMARY_LLM_PROVIDER` | `"ollama"` | `SUMMARY_LLM_PROVIDER` | `llm_provider.py` |
| `TRANSLATION_LLM_PROVIDER` | `"ollama"` | `TRANSLATION_LLM_PROVIDER` | `llm_provider.py` |
| `QUIZ_LLM_PROVIDER` | `"ollama"` | `QUIZ_LLM_PROVIDER` | `llm_provider.py` |
| `RAG_LLM_PROVIDER` | `"ollama"` | `RAG_LLM_PROVIDER` | `llm_provider.py` |
| `ALLOW_OLLAMA_TO_LOCAL_FALLBACK` | `false` | `ALLOW_OLLAMA_TO_LOCAL_FALLBACK` | `llm_provider.py` |
| `OLLAMA_BASE_URL` | `"http://127.0.0.1:11434"` | `OLLAMA_BASE_URL` | `llm_provider.py` |
| `OLLAMA_MODEL` | `"qwen3.5:4b"` | `OLLAMA_MODEL` | `llm_provider.py` |
| `OLLAMA_KEEP_ALIVE` | `"10m"` | `OLLAMA_KEEP_ALIVE` | `llm_provider.py` |
| `OLLAMA_TIMEOUT_SEC` | `300` | `OLLAMA_TIMEOUT_SEC` | `llm_provider.py` |
| `OLLAMA_THINK` | `false` | `OLLAMA_THINK` | `llm_provider.py` |
