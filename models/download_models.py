"""Download every model this project uses into the standard local caches.

Usage:  python models/download_models.py [--skip-ollama] [--skip-whisper-extra]
"""
import argparse
import shutil
import subprocess
import sys

HF_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "openai/whisper-small",
]
WHISPER_DEFAULT = ["base"]
WHISPER_EXTRA = ["small", "medium"]
OLLAMA_MODELS = ["qwen2.5:1.5b"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ollama", action="store_true")
    ap.add_argument("--skip-whisper-extra", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    import whisper

    for repo in HF_MODELS:
        print(f"[hf] {repo}")
        snapshot_download(repo_id=repo)

    names = WHISPER_DEFAULT + ([] if args.skip_whisper_extra else WHISPER_EXTRA)
    for name in names:
        print(f"[whisper] {name}")
        whisper.load_model(name, device="cpu")

    if not args.skip_ollama:
        if shutil.which("ollama") is None:
            print("ollama CLI not found; install from https://ollama.com/download then re-run.")
            return 1
        for tag in OLLAMA_MODELS:
            print(f"[ollama] {tag}")
            subprocess.run(["ollama", "pull", tag], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
