"""
app/services/local_llm_worker.py -- Isolated subprocess runner & shared HF Transformers helper
for LocalLLMProvider (Qwen2.5-0.5B).

Running HF/PyTorch CPU inference inside a managed child process guarantees 100% OS-level release
of Working Set, Private Commit, and Windows Segment Heap arenas when LocalLLMProvider.unload()
terminates the worker between notes/summary and Ollama quiz/RAG phases.
"""

from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import sys
from typing import Any


def load_hf_causal_lm(model_name: str, device: str) -> tuple[Any, Any]:
    """Load AutoTokenizer and AutoModelForCausalLM identically to the original pipeline."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) - 1)))
    use_cuda = (device != "cpu") and torch.cuda.is_available()
    target_dtype = torch.float16 if use_cuda else torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=target_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=target_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    model.eval()
    if use_cuda:
        model.to(device)
    return tokenizer, model


def run_hf_generate(
    model: Any,
    tokenizer: Any,
    model_name: str,
    device: str,
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """Execute HF Transformers chat formatting and generation without modifying generation parameters."""
    import torch

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        formatted_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        formatted_input = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"

    inputs = tokenizer(
        [formatted_input],
        return_tensors="pt",
    )
    if device != "cpu" and torch.cuda.is_available():
        inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0.05
    eos_ids = [tokenizer.eos_token_id]
    if "qwen" in model_name.lower():
        eos_ids.extend([151645, 151643])
    eos_ids = [eid for eid in set(eos_ids) if eid is not None]

    gen_kwargs = {
        "max_new_tokens": min(max_tokens, 4096),
        "do_sample": do_sample,
        "repetition_penalty": 1.12,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": eos_ids[0] if len(eos_ids) == 1 else eos_ids,
    }
    if do_sample:
        gen_kwargs["temperature"] = max(temperature, 0.05)
        gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_kwargs,
        )

    gen_tokens = outputs[0][input_len:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def _serve_stdio(model_name: str, device: str) -> int:
    # Reserve sys.__stdout__ exclusively for JSON-lines IPC; route library prints/progress to stderr.
    ipc_out = sys.stdout
    sys.stdout = sys.stderr

    def send(payload: dict[str, Any]) -> None:
        ipc_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
        ipc_out.flush()

    try:
        tokenizer, model = load_hf_causal_lm(model_name, device)
    except Exception as exc:
        send({"type": "init_error", "error": str(exc)})
        return 1

    send({"type": "ready", "pid": os.getpid()})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            send({"ok": False, "error": f"Invalid IPC request: {exc}"})
            continue

        cmd = req.get("cmd")
        if cmd == "ping":
            send({"ok": True, "pid": os.getpid()})
        elif cmd == "shutdown":
            send({"ok": True})
            break
        elif cmd == "generate":
            try:
                text = run_hf_generate(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    device=device,
                    prompt=req.get("prompt", ""),
                    system_prompt=req.get("system_prompt", ""),
                    max_tokens=int(req.get("max_tokens", 256)),
                    temperature=float(req.get("temperature", 0.1)),
                )
                send({"ok": True, "text": text})
            except Exception as exc:
                send({"ok": False, "error": str(exc)})
        else:
            send({"ok": False, "error": f"Unknown cmd: {cmd}"})

    return 0


def main() -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Isolated LocalLLMProvider worker process")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    raise SystemExit(_serve_stdio(args.model, args.device))


if __name__ == "__main__":
    main()
