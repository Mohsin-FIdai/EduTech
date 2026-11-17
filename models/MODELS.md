# Models used by this project

Model weights are **not stored in this repository** (about 4.3 GB on the original machine, all
publicly downloadable). Run `python models/download_models.py` to fetch them into the default caches.

| Purpose | Model | Source | Config / code reference |
|---|---|---|---|
| Speech-to-text (default) | Whisper `base` | openai-whisper (`base.pt`, ~139 MB) | `DEFAULT_WHISPER_MODEL` in `app/config.py` |
| Speech-to-text (optional sizes) | Whisper `small`, `medium` | openai-whisper | `SUPPORTED_WHISPER_MODELS` in `app/config.py` |
| Local LLM (notes, summary, translation fallback) | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face | `LOCAL_LLM_MODEL` in `app/config.py` |
| Ollama LLM (quiz generation, RAG Q&A) | `qwen2.5:1.5b` | Ollama | `OLLAMA_MODEL` in `app/config.py` |
| Embeddings / semantic search | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Hugging Face | `MODEL_NAME` in `app/services/embedding_service.py` |
| In the HF cache, not referenced by `app/` | `sentence-transformers/all-MiniLM-L6-v2`, `openai/whisper-small` (Transformers format) | Hugging Face | present on the dev PC only; included in the download script for completeness |

Original cache locations on the development PC:

- `C:\Users\<user>\.cache\whisper` (Whisper `.pt` files)
- `C:\Users\<user>\.cache\huggingface\hub` (Qwen, MiniLM, whisper-small)
- `C:\Users\<user>\.ollama\models` (Ollama blobs)

Ollama itself must be installed separately from https://ollama.com/download (the installer is not committed).
