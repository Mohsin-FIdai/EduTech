"""
app/services/llm_provider.py -- Provider-independent LLM abstraction layer.

Defines:
  BaseLLMProvider (ABC)
  LocalLLMProvider (Transformers AutoModelForCausalLM singleton)
  OllamaLLMProvider (Optional local Ollama HTTP API provider)
  CloudLLMProvider (Pluggable Cloud LLM stub/adapter)
  get_llm_provider() (Factory function)
"""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import ctypes.wintypes as _wintypes
import gc
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _ensure_lifecycle_logger_handler() -> None:
    """Ensure [LLM_LIFECYCLE] INFO logs are visible under Uvicorn without interfering with pytest caplog."""
    if logger.handlers:
        return
    if os.getenv("TESTING") == "1" or ("pytest" in sys.modules):
        return
    if not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)


def _query_nt_process_mem(k32: Any, handle: Any) -> Optional[tuple[float, float, float]]:
    class _PMC_EX(ctypes.Structure):
        _fields_ = [
            ("cb", _wintypes.DWORD),
            ("PageFaultCount", _wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    k32.K32GetProcessMemoryInfo.argtypes = [
        _wintypes.HANDLE,
        ctypes.POINTER(_PMC_EX),
        _wintypes.DWORD,
    ]
    k32.K32GetProcessMemoryInfo.restype = _wintypes.BOOL
    pmc = _PMC_EX()
    pmc.cb = ctypes.sizeof(pmc)
    if k32.K32GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
        return (
            round(pmc.WorkingSetSize / 1048576, 1),
            round(pmc.PeakWorkingSetSize / 1048576, 1),
            round(pmc.PrivateUsage / 1048576, 1),
        )
    return None


def _trim_windows_working_set() -> None:
    if os.name != "nt":
        return
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = _wintypes.HANDLE
        k32.K32EmptyWorkingSet.argtypes = [_wintypes.HANDLE]
        k32.K32EmptyWorkingSet.restype = _wintypes.BOOL
        k32.K32EmptyWorkingSet(k32.GetCurrentProcess())
    except Exception:
        pass


def get_process_memory_snapshot(extra_pid: Optional[int] = None) -> dict[str, Any]:
    """
    Measure actual process Working Set, Private commit, Peak WS, active local worker memory (if any),
    and System Available RAM (MB) without third-party dependencies.
    """
    snap: dict[str, Any] = {
        "ws_mb": None,
        "peak_ws_mb": None,
        "private_mb": None,
        "worker_pid": None,
        "worker_ws_mb": None,
        "worker_priv_mb": None,
        "total_ws_mb": None,
        "total_priv_mb": None,
        "sys_avail_mb": None,
        "sys_used_pct": None,
    }
    if extra_pid is None:
        inst = getattr(LocalLLMProvider, "_instance", None) if "LocalLLMProvider" in globals() else None
        proc = getattr(inst, "_worker_proc", None) if inst is not None else None
        if proc is not None and proc.poll() is None:
            extra_pid = proc.pid

    if os.name == "nt":
        try:
            class _MEMSTAT(ctypes.Structure):
                _fields_ = [
                    ("dwLength", _wintypes.DWORD),
                    ("dwMemoryLoad", _wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = _wintypes.HANDLE
            k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMSTAT)]
            k32.GlobalMemoryStatusEx.restype = _wintypes.BOOL

            cur_mem = _query_nt_process_mem(k32, k32.GetCurrentProcess())
            if cur_mem is not None:
                snap["ws_mb"], snap["peak_ws_mb"], snap["private_mb"] = cur_mem
                snap["total_ws_mb"] = snap["ws_mb"]
                snap["total_priv_mb"] = snap["private_mb"]

            if extra_pid:
                k32.OpenProcess.argtypes = [_wintypes.DWORD, _wintypes.BOOL, _wintypes.DWORD]
                k32.OpenProcess.restype = _wintypes.HANDLE
                k32.CloseHandle.argtypes = [_wintypes.HANDLE]
                k32.CloseHandle.restype = _wintypes.BOOL
                h_child = k32.OpenProcess(0x1010, False, int(extra_pid))
                if h_child:
                    try:
                        child_mem = _query_nt_process_mem(k32, h_child)
                        if child_mem is not None:
                            snap["worker_pid"] = int(extra_pid)
                            snap["worker_ws_mb"] = child_mem[0]
                            snap["worker_priv_mb"] = child_mem[2]
                            if snap["ws_mb"] is not None:
                                snap["total_ws_mb"] = round(snap["ws_mb"] + child_mem[0], 1)
                            if snap["private_mb"] is not None:
                                snap["total_priv_mb"] = round(snap["private_mb"] + child_mem[2], 1)
                    finally:
                        k32.CloseHandle(h_child)

            ms = _MEMSTAT()
            ms.dwLength = ctypes.sizeof(ms)
            if k32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                snap["sys_avail_mb"] = round(ms.ullAvailPhys / 1048576, 1)
                snap["sys_used_pct"] = int(ms.dwMemoryLoad)
        except Exception:
            pass
    return snap


class LLMLifecycleManager:
    """
    Process-wide coordinator enforcing sequential residency between Transformers (LocalLLMProvider)
    and Ollama (OllamaLLMProvider), preventing duplicate loading, premature unloading while
    active leases exist, and deadlocks across nested calls.
    """

    def __init__(self) -> None:
        self._exec_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._active_leases: dict[str, int] = {"local": 0, "ollama": 0}
        self._thread_local = threading.local()

    def current_thread_holds_lock(self) -> bool:
        return getattr(self._thread_local, "exec_depth", 0) > 0

    def _enter_exec(self) -> None:
        self._exec_lock.acquire()
        self._thread_local.exec_depth = getattr(self._thread_local, "exec_depth", 0) + 1

    def _exit_exec(self) -> None:
        depth = getattr(self._thread_local, "exec_depth", 0)
        if depth > 0:
            self._thread_local.exec_depth = depth - 1
        self._exec_lock.release()

    def active_leases_snapshot(self) -> dict[str, int]:
        with self._state_lock:
            return dict(self._active_leases)

    def _prepare_for_kind(self, kind: str, feature: str) -> None:
        """
        Ensure mutual exclusion between heavy providers before activation/loading.
        Must only be invoked while _exec_lock is held by the current thread.
        """
        if kind == "local":
            local_p = LocalLLMProvider.get_instance()
            if not local_p.is_loaded():
                ollama_p = OllamaLLMProvider.get_instance()
                evicted = ollama_p.evict_loaded_models(reason=f"prepare_local:{feature}")
                if evicted:
                    logger.info(
                        f"[LLM_LIFECYCLE] action=evict_ollama feature={feature} "
                        f"models={evicted} mem={get_process_memory_snapshot()}"
                    )
        elif kind == "ollama":
            local_p = LocalLLMProvider.get_instance()
            if local_p.is_loaded():
                with self._state_lock:
                    local_leases = self._active_leases["local"]
                if local_leases == 0:
                    local_p.unload(reason=f"switch_to_ollama:{feature}")
                else:
                    logger.warning(
                        f"[LLM_LIFECYCLE] action=skip_unload_active_local feature={feature} "
                        f"active_local_leases={local_leases}"
                    )

    @contextlib.contextmanager
    def lease(
        self,
        kind: str,
        feature: str = "default",
        release_local_on_exit: bool = False,
    ) -> Iterator[None]:
        """Acquire serialized execution + active refcount lease for `kind` ('local' or 'ollama')."""
        _ensure_lifecycle_logger_handler()
        self._enter_exec()
        lease_counted = False
        try:
            self._prepare_for_kind(kind, feature)
            with self._state_lock:
                self._active_leases[kind] = self._active_leases.get(kind, 0) + 1
                lease_counted = True
            logger.info(
                f"[LLM_LIFECYCLE] action=activate provider={kind} feature={feature} "
                f"leases={self.active_leases_snapshot()} mem={get_process_memory_snapshot()}"
            )
            yield
        finally:
            should_unload_local = False
            if lease_counted:
                with self._state_lock:
                    cur = max(0, self._active_leases.get(kind, 0) - 1)
                    self._active_leases[kind] = cur
                    if self._active_leases.get("local", 0) == 0 and (
                        release_local_on_exit or LocalLLMProvider.get_instance().is_loaded() and kind != "local"
                    ):
                        should_unload_local = release_local_on_exit or LocalLLMProvider.get_instance().is_loaded()
            if should_unload_local:
                LocalLLMProvider.get_instance().unload(reason=f"session_complete:{feature}")
            self._exit_exec()

    @contextlib.contextmanager
    def feature_session(self, feature: str, release_on_exit: Optional[bool] = None) -> Iterator[None]:
        """Acquire lifecycle session matching the configured provider for `feature`."""
        import sys
        from app.config import ENABLE_LOCAL_LLM

        in_test_mode = os.getenv("TESTING") == "1" or ("pytest" in sys.modules)
        if in_test_mode and not ENABLE_LOCAL_LLM:
            yield
            return

        provider_name = resolve_provider_name(feature)
        if provider_name == "cloud":
            yield
            return

        kind = "ollama" if provider_name == "ollama" else "local"
        if kind == "ollama" and not OllamaLLMProvider.get_instance().is_available():
            from app.config import ALLOW_OLLAMA_TO_LOCAL_FALLBACK
            from roman_urdu_nlp.utils import env_flag
            if env_flag("ALLOW_OLLAMA_TO_LOCAL_FALLBACK", default=ALLOW_OLLAMA_TO_LOCAL_FALLBACK):
                kind = "local"
            else:
                yield
                return

        if release_on_exit is None:
            is_cuda = str(getattr(LocalLLMProvider.get_instance(), "device", "cpu")).startswith("cuda")
            release_on_exit = False if is_cuda else (feature in ("notes", "notes_summary", "summary", "quiz"))

        with self.lease(kind, feature=feature, release_local_on_exit=bool(release_on_exit)):
            yield


_LIFECYCLE = LLMLifecycleManager()


def _cleanup_local_worker_on_exit() -> None:
    inst = getattr(LocalLLMProvider, "_instance", None) if "LocalLLMProvider" in globals() else None
    if inst is not None:
        try:
            inst._terminate_worker_locked()
        except Exception:
            pass


atexit.register(_cleanup_local_worker_on_exit)


def get_llm_lifecycle_manager() -> LLMLifecycleManager:
    return _LIFECYCLE


@contextlib.contextmanager
def llm_lifecycle_session(feature: str, release_on_exit: Optional[bool] = None) -> Iterator[None]:
    """Convenience context manager for scoped sequential LLM lifecycle management."""
    with _LIFECYCLE.feature_session(feature, release_on_exit=release_on_exit):
        yield


class LLMException(Exception):
    """Base exception for LLM provider errors."""
    pass


class BaseLLMProvider(ABC):
    """Abstract base class for all language model providers."""

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1
    ) -> str:
        """Generate unstructured text from prompt."""
        raise NotImplementedError

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: Optional[dict[str, Any]] = None,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1
    ) -> dict[str, Any]:
        """Generate structured JSON dictionary adhering to optional schema."""
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider dependencies and model weights can be loaded."""
        raise NotImplementedError

    def complete(self, prompt: str, max_tokens: int = 2048) -> str:
        """Convenience completion method."""
        return self.generate_text(prompt=prompt, max_tokens=max_tokens)


class LocalLLMProvider(BaseLLMProvider):
    """
    Local CPU-friendly instruction-tuned language model provider.
    Uses Hugging Face Transformers AutoModelForCausalLM (default: Qwen/Qwen2.5-0.5B-Instruct).
    Fully offline, zero external API calls.
    """
    _instance: Optional[LocalLLMProvider] = None
    _lock = threading.Lock()
    _gen_lock = threading.Lock()  # leaf inference lock; outer coordinator lock is LLMLifecycleManager._exec_lock
    _tokenizer: Optional[Any] = None
    _model: Optional[Any] = None
    _worker_proc: Optional[subprocess.Popen] = None
    _init_failed: bool = False
    _init_error: Optional[str] = None
    _init_failed_at: float = 0.0

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        from app.config import LOCAL_LLM_MODEL, LOCAL_LLM_DEVICE
        self.model_name = model_name or LOCAL_LLM_MODEL
        self.device = device or LOCAL_LLM_DEVICE

    @classmethod
    def get_instance(cls) -> LocalLLMProvider:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _use_subprocess_worker(self) -> bool:
        if self._model is not None and self._tokenizer is not None:
            return False
        default_val = "false" if str(self.device).startswith("cuda") else "true"
        raw = os.getenv("LOCAL_LLM_SUBPROCESS", default_val).strip().lower()
        return raw not in ("0", "false", "no", "off")

    def is_loaded(self) -> bool:
        proc = self._worker_proc
        return (self._model is not None) or (proc is not None and proc.poll() is None)

    def _terminate_worker_locked(self) -> None:
        proc = self._worker_proc
        self._worker_proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write('{"cmd":"shutdown"}\n')
                proc.stdin.flush()
                proc.wait(timeout=3.0)
        except Exception:
            pass
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

    def unload(self, reason: str = "manual") -> bool:
        """
        Safely unload PyTorch model weights and tokenizer (terminating worker subprocess if active)
        while keeping singleton and config stable. Never unloads if active local lifecycle leases > 0.
        """
        if _LIFECYCLE.active_leases_snapshot().get("local", 0) > 0:
            logger.warning(
                f"[LLM_LIFECYCLE] action=unload_blocked provider=local reason={reason} "
                f"leases={_LIFECYCLE.active_leases_snapshot()}"
            )
            return False

        with self._lock:
            proc = self._worker_proc
            worker_alive = proc is not None and proc.poll() is None
            if self._model is None and self._tokenizer is None and proc is None:
                return False
            before = get_process_memory_snapshot(extra_pid=proc.pid if worker_alive else None)
            self._terminate_worker_locked()
            model_ref = self._model
            tok_ref = self._tokenizer
            had_in_process = model_ref is not None or tok_ref is not None
            self._model = None
            self._tokenizer = None
            self._init_failed = False
            self._init_error = None
            del model_ref, tok_ref
            if had_in_process:
                gc.collect()
                _trim_windows_working_set()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            after = get_process_memory_snapshot()
            b_ws = before.get("total_ws_mb") if before.get("total_ws_mb") is not None else before.get("ws_mb")
            a_ws = after.get("total_ws_mb") if after.get("total_ws_mb") is not None else after.get("ws_mb")
            b_priv = before.get("total_priv_mb") if before.get("total_priv_mb") is not None else before.get("private_mb")
            a_priv = after.get("total_priv_mb") if after.get("total_priv_mb") is not None else after.get("private_mb")
            freed_ws = round(b_ws - a_ws, 1) if b_ws is not None and a_ws is not None else None
            freed_priv = round(b_priv - a_priv, 1) if b_priv is not None and a_priv is not None else None
            avail_delta = (
                round(after["sys_avail_mb"] - before["sys_avail_mb"], 1)
                if before.get("sys_avail_mb") is not None and after.get("sys_avail_mb") is not None
                else None
            )
            _ensure_lifecycle_logger_handler()
            logger.info(
                f"[LLM_LIFECYCLE] action=unload provider=local model={self.model_name} reason={reason} "
                f"before={before} after={after} freed_ws_mb={freed_ws} freed_priv_mb={freed_priv} "
                f"sys_avail_delta_mb={avail_delta}"
            )
            return True

    def _load(self) -> None:
        """Lazy thread-safe loader for local model and tokenizer (isolated worker by default)."""
        if self.is_loaded():
            return
        if self._init_failed and (time.monotonic() - self._init_failed_at) < 15.0:
            return

        with self._lock:
            if self.is_loaded():
                return
            if self._init_failed and (time.monotonic() - self._init_failed_at) < 15.0:
                return
            self._init_failed = False
            self._init_error = None
            t0 = time.perf_counter()
            before = get_process_memory_snapshot()
            _ensure_lifecycle_logger_handler()
            try:
                if self._use_subprocess_worker():
                    logger.info(
                        f"[LLM_LIFECYCLE] action=load_start provider=local mode=subprocess "
                        f"model={self.model_name} device={self.device} mem={before}"
                    )
                    project_root = str(Path(__file__).resolve().parents[2])
                    env = os.environ.copy()
                    env["PYTHONIOENCODING"] = "utf-8"
                    env["PYTHONUTF8"] = "1"
                    proc = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "app.services.local_llm_worker",
                            "--model",
                            self.model_name,
                            "--device",
                            self.device,
                        ],
                        cwd=project_root,
                        env=env,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=None,
                        text=True,
                        encoding="utf-8",
                        bufsize=1,
                    )
                    ready_line = proc.stdout.readline() if proc.stdout is not None else ""
                    if not ready_line:
                        rc = proc.poll()
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        raise RuntimeError(f"Worker exited before ready (rc={rc})")
                    ready_msg = json.loads(ready_line)
                    if ready_msg.get("type") != "ready":
                        err = ready_msg.get("error") or str(ready_msg)
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        raise RuntimeError(f"Worker init failed: {err}")
                    self._worker_proc = proc
                    after = get_process_memory_snapshot(extra_pid=proc.pid)
                    load_ms = round((time.perf_counter() - t0) * 1000, 1)
                    logger.info(
                        f"[LLM_LIFECYCLE] action=load_done provider=local mode=subprocess "
                        f"worker_pid={proc.pid} model={self.model_name} "
                        f"elapsed_ms={load_ms} before={before} after={after}"
                    )
                    return

                from app.services.local_llm_worker import load_hf_causal_lm

                logger.info(
                    f"[LLM_LIFECYCLE] action=load_start provider=local mode=in_process "
                    f"model={self.model_name} device={self.device} mem={before}"
                )
                self._tokenizer, self._model = load_hf_causal_lm(self.model_name, self.device)
                after = get_process_memory_snapshot()
                load_ms = round((time.perf_counter() - t0) * 1000, 1)
                logger.info(
                    f"[LLM_LIFECYCLE] action=load_done provider=local mode=in_process model={self.model_name} "
                    f"elapsed_ms={load_ms} before={before} after={after}"
                )
            except Exception as e:
                self._terminate_worker_locked()
                self._init_failed = True
                self._init_failed_at = time.monotonic()
                self._init_error = str(e)
                logger.error(f"Failed to load local LLM {self.model_name}: {e}")
                raise LLMException(f"Failed to initialize local LLM '{self.model_name}': {e}") from e

    def is_available(self) -> bool:
        try:
            if self.is_loaded():
                return True
            with _LIFECYCLE.lease("local", feature="is_available", release_local_on_exit=False):
                self._load()
                return self.is_loaded() and not self._init_failed
        except Exception:
            return False

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 256,
        temperature: float = 0.1
    ) -> str:
        from app.config import ENABLE_LOCAL_LLM

        in_test_stub = (os.getenv("TESTING") == "1" or ("pytest" in sys.modules)) and not ENABLE_LOCAL_LLM
        if _LIFECYCLE.current_thread_holds_lock() or (in_test_stub and self._model is not None):
            with self._gen_lock:
                return self._generate_text_locked(prompt, system_prompt, max_tokens, temperature)
        with _LIFECYCLE.lease("local", feature="generate_text", release_local_on_exit=False):
            with self._gen_lock:
                return self._generate_text_locked(prompt, system_prompt, max_tokens, temperature)

    def _generate_text_locked(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 256,
        temperature: float = 0.1
    ) -> str:
        self._load()
        t_inf0 = time.perf_counter()

        if self._model is not None and self._tokenizer is not None:
            from app.services.local_llm_worker import run_hf_generate

            response = run_hf_generate(
                model=self._model,
                tokenizer=self._tokenizer,
                model_name=self.model_name,
                device=self.device,
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response = self._detect_repetition(response)
            logger.debug(
                f"[LLM_LIFECYCLE] action=infer_done provider=local model={self.model_name} "
                f"elapsed_ms={round((time.perf_counter() - t_inf0) * 1000, 1)}"
            )
            return response

        proc = self._worker_proc
        if proc is None or proc.poll() is not None or proc.stdin is None or proc.stdout is None:
            self._terminate_worker_locked()
            raise LLMException("Local LLM worker is not running.")

        payload = json.dumps(
            {
                "cmd": "generate",
                "prompt": prompt,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            ensure_ascii=False,
        )
        try:
            proc.stdin.write(payload + "\n")
            proc.stdin.flush()
            resp_line = proc.stdout.readline()
        except Exception as e:
            self._terminate_worker_locked()
            self._init_failed = True
            self._init_failed_at = time.monotonic()
            self._init_error = str(e)
            raise LLMException(f"Local LLM worker IPC failed: {e}") from e

        if not resp_line:
            rc = proc.poll()
            self._terminate_worker_locked()
            self._init_failed = True
            self._init_failed_at = time.monotonic()
            self._init_error = f"Worker exited mid-generation (rc={rc})"
            raise LLMException(self._init_error)

        resp = json.loads(resp_line)
        if not resp.get("ok"):
            raise LLMException(f"Local LLM generation failed: {resp.get('error', 'unknown error')}")

        response = self._detect_repetition(str(resp.get("text", "")))
        logger.debug(
            f"[LLM_LIFECYCLE] action=infer_done provider=local model={self.model_name} "
            f"elapsed_ms={round((time.perf_counter() - t_inf0) * 1000, 1)}"
        )
        return response

    def generate_structured(
        self,
        prompt: str,
        schema: Optional[dict[str, Any]] = None,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1
    ) -> dict[str, Any]:
        """Generate structured JSON adhering to requested schema."""
        schema_instruction = ""
        if schema:
            schema_str = json.dumps(schema, indent=2)
            schema_instruction = (
                f"\nCRITICAL INSTRUCTION: You MUST output ONLY valid JSON adhering strictly to this schema:\n"
                f"{schema_str}\n"
                f"Do not write conversational preamble or postscript. Output ONLY raw JSON inside {{ ... }}."
            )

        full_system = f"{system_prompt}\n{schema_instruction}".strip()
        raw_output = self.generate_text(
            prompt=prompt,
            system_prompt=full_system,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return self._extract_json(raw_output)

    @staticmethod
    def _detect_repetition(
        text: str,
        ngram_size: Optional[int] = None,
        max_repeats: Optional[int] = None,
        line_threshold: Optional[float] = None,
    ) -> str:
        """
        Detect and truncate repetition loops in LLM output.

        Two complementary checks:
        1. **N-gram repeat**: if any n-gram of *ngram_size* tokens appears more
           than *max_repeats* times, the output is truncated just before the
           (max_repeats+1)-th occurrence of that n-gram.
        2. **Line-level repeat**: if the fraction of non-empty lines that are
           exact duplicates of a previously seen line exceeds *line_threshold*,
           the output is truncated at the first repeated line.

        Config defaults come from app.config so callers and tests can override
        via env vars without touching code.
        """
        if not text:
            return text

        # Resolve defaults lazily so tests can monkeypatch config or pass overrides
        try:
            from app.config import (
                LLM_REPETITION_NGRAM_SIZE,
                LLM_REPETITION_MAX_REPEATS,
                LLM_REPETITION_LINE_THRESHOLD,
            )
        except Exception:  # pragma: no cover – config unavailable in unit tests
            LLM_REPETITION_NGRAM_SIZE = 4
            LLM_REPETITION_MAX_REPEATS = 3
            LLM_REPETITION_LINE_THRESHOLD = 0.4

        n = ngram_size if ngram_size is not None else LLM_REPETITION_NGRAM_SIZE
        max_r = max_repeats if max_repeats is not None else LLM_REPETITION_MAX_REPEATS
        line_thr = line_threshold if line_threshold is not None else LLM_REPETITION_LINE_THRESHOLD

        # ── 1. Line-level duplicate detection ────────────────────────
        lines = text.splitlines()
        seen_lines: dict[str, int] = {}
        dup_count = 0
        truncate_at_char: Optional[int] = None
        char_pos = 0
        for line in lines:
            stripped = line.strip()
            line_len = len(line) + 1  # +1 for the newline
            if stripped:
                if stripped in seen_lines:
                    dup_count += 1
                    non_empty_total = sum(1 for l in lines if l.strip())
                    if non_empty_total > 0 and dup_count / non_empty_total >= line_thr:
                        truncate_at_char = char_pos
                        logger.warning(
                            "LLM output line-loop detected (%.0f%% duplicate lines); "
                            "truncating at char %d.",
                            100 * dup_count / non_empty_total,
                            char_pos,
                        )
                        break
                else:
                    seen_lines[stripped] = char_pos
            char_pos += line_len

        if truncate_at_char is not None:
            text = text[:truncate_at_char].rstrip()
            if not text:
                return "[output truncated: repetition loop detected]"
            return text

        # ── 2. N-gram token-level detection ──────────────────────────
        words = text.split()
        if len(words) < n * (max_r + 1):
            # Too short to possibly contain max_r+1 distinct occurrences
            return text

        ngram_positions: dict[tuple, list[int]] = {}
        for i in range(len(words) - n + 1):
            ng = tuple(words[i: i + n])
            ngram_positions.setdefault(ng, []).append(i)

        # Find the earliest position where a repeat exceeds the limit
        earliest_cutoff: Optional[int] = None
        for ng, positions in ngram_positions.items():
            if len(positions) > max_r:
                # The (max_r+1)-th occurrence starts at positions[max_r]
                cutoff_word_idx = positions[max_r]
                if earliest_cutoff is None or cutoff_word_idx < earliest_cutoff:
                    earliest_cutoff = cutoff_word_idx

        if earliest_cutoff is not None and earliest_cutoff > 0:
            truncated = " ".join(words[:earliest_cutoff]).rstrip()
            logger.warning(
                "LLM output n-gram loop detected (ngram=%r repeated >%d times); "
                "truncating after word %d.",
                " ".join(words[earliest_cutoff: earliest_cutoff + n]),
                max_r,
                earliest_cutoff,
            )
            return truncated if truncated else "[output truncated: repetition loop detected]"

        return text

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract and parse JSON from text, handling markdown fences, unclosed brackets, and partial outputs."""
        cleaned = text.strip()
        # 1. Remove markdown code fences ```json ... ```
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)(?:```|$)", cleaned, re.IGNORECASE)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        # 2. Try direct JSON parse
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # 3. Find outermost JSON object { ... }
        start = cleaned.find("{")
        if start != -1:
            snippet = cleaned[start:]
            # If closed properly or ends with an inner object brace:
            end = snippet.rfind("}")
            if end != -1:
                candidate = snippet[: end + 1]
                # Balance any open brackets or braces (e.g. truncated arrays of objects)
                open_brackets = max(0, candidate.count("[") - candidate.count("]"))
                open_braces = max(0, candidate.count("{") - candidate.count("}"))
                candidate_balanced = candidate + ("]" * open_brackets) + ("}" * open_braces)
                try:
                    return json.loads(candidate_balanced)
                except Exception:
                    repaired = re.sub(r",\s*([\]}])", r"\1", candidate_balanced)
                    try:
                        return json.loads(repaired)
                    except Exception:
                        pass

            # 4. Attempt partial-JSON repair for truncated responses
            repaired_snippet = snippet
            # If odd number of unescaped quotes, add closing quote
            num_quotes = len(re.findall(r'(?<!\\)"', repaired_snippet))
            if num_quotes % 2 != 0:
                repaired_snippet += '"'

            # Clean trailing partial keys e.g. `, "some_key": "` or `, "some_key`
            repaired_snippet = re.sub(r',\s*"[^"]*"?\s*:?\s*("[^"]*)?$', '', repaired_snippet)

            # Balance open brackets and braces
            open_brackets = max(0, repaired_snippet.count("[") - repaired_snippet.count("]"))
            open_braces = max(0, repaired_snippet.count("{") - repaired_snippet.count("}"))
            balanced = repaired_snippet + ("]" * open_brackets) + ("}" * open_braces)

            try:
                return json.loads(balanced)
            except Exception:
                balanced = re.sub(r",\s*([\]}])", r"\1", balanced)
                try:
                    return json.loads(balanced)
                except Exception:
                    pass

        # 4b. Recover complete JSON objects inside a truncated "quizzes" array
        if '"quizzes"' in cleaned:
            q_arr_pos = cleaned.find('"quizzes"')
            arr_start = cleaned.find("[", q_arr_pos)
            if arr_start != -1:
                recovered_quizzes = []
                depth = 0
                in_str = False
                esc = False
                obj_start = -1
                for idx in range(arr_start + 1, len(cleaned)):
                    ch = cleaned[idx]
                    if esc:
                        esc = False
                        continue
                    if ch == "\\" and in_str:
                        esc = True
                        continue
                    if ch == '"':
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if ch == "{":
                        if depth == 0:
                            obj_start = idx
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0 and obj_start != -1:
                            try:
                                recovered_quizzes.append(json.loads(cleaned[obj_start: idx + 1]))
                            except Exception:
                                pass
                            obj_start = -1
                if recovered_quizzes:
                    return {"quizzes": recovered_quizzes}

        # 5. Regex-based key-value extraction fallback
        extracted: dict[str, Any] = {}
        for key in ["subject", "theme", "overview", "title", "explanation"]:
            m = re.search(rf'"{key}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', cleaned, re.IGNORECASE)
            if m:
                try:
                    extracted[key] = m.group(1).encode("utf-8").decode("unicode_escape", errors="ignore")
                except Exception:
                    extracted[key] = m.group(1)

        for list_key in ["key_points", "examples", "formulas"]:
            list_match = re.search(rf'"{list_key}"\s*:\s*\[([\s\S]*?)\]', cleaned, re.IGNORECASE)
            if list_match:
                items = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', list_match.group(1))
                if items:
                    extracted[list_key] = [it.strip() for it in items if len(it.strip()) > 3]

        sec_match = re.search(r'"sections"\s*:\s*\[([\s\S]*?)\]', cleaned, re.IGNORECASE)
        if sec_match:
            sec_blocks = re.findall(r'\{([^{}]+)\}', sec_match.group(1))
            sections_list = []
            for b in sec_blocks:
                t_m = re.search(r'"title"\s*:\s*"([^"]+)"', b)
                s_m = re.search(r'"start_segment_id"\s*:\s*(\d+)', b)
                e_m = re.search(r'"end_segment_id"\s*:\s*(\d+)', b)
                if t_m and s_m and e_m:
                    sections_list.append({
                        "title": t_m.group(1),
                        "start_segment_id": int(s_m.group(1)),
                        "end_segment_id": int(e_m.group(1))
                    })
            if sections_list:
                extracted["sections"] = sections_list

        if extracted.get("title") or extracted.get("subject") or extracted.get("overview") or extracted.get("sections"):
            return extracted

        logger.warning(f"Could not parse structured JSON from LLM output: {text[:200]}...")
        return {"raw_response": text, "parse_error": True}


class CloudLLMProvider(BaseLLMProvider):
    """
    Pluggable Cloud LLM Provider (e.g. Gemini, OpenAI, Anthropic).
    Ready for zero-rewrite cloud extension when API keys are available.
    """
    def __init__(self, provider_name: str = "gemini"):
        self.provider_name = provider_name
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1
    ) -> str:
        if not self.is_available():
            raise LLMException("Cloud LLM API key not found.")
        # Future cloud implementation: google.generativeai call
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_prompt if system_prompt else None)
        resp = model.generate_content(prompt)
        return resp.text.strip()

    def generate_structured(
        self,
        prompt: str,
        schema: Optional[dict[str, Any]] = None,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1
    ) -> dict[str, Any]:
        text = self.generate_text(prompt, system_prompt, max_tokens, temperature)
        return LocalLLMProvider._extract_json(text)


class OllamaLLMProvider(BaseLLMProvider):
    """
    Optional local Ollama HTTP API provider (e.g. qwen2.5:1.5b or qwen2.5:0.5b).
    Runs independently of PyTorch Transformers and coordinates sequential memory residency
    through LLMLifecycleManager.
    """
    _instance: Optional[OllamaLLMProvider] = None
    _lock = threading.Lock()
    _gen_lock = threading.Lock()

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        keep_alive: Optional[str] = None,
        timeout_sec: Optional[int] = None,
    ):
        from app.config import (
            OLLAMA_BASE_URL,
            OLLAMA_MODEL,
            OLLAMA_KEEP_ALIVE,
            OLLAMA_TIMEOUT_SEC,
        )
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)).rstrip("/")
        self.keep_alive = keep_alive or os.getenv("OLLAMA_KEEP_ALIVE", OLLAMA_KEEP_ALIVE)
        self.timeout_sec = timeout_sec or int(os.getenv("OLLAMA_TIMEOUT_SEC", str(OLLAMA_TIMEOUT_SEC)))

    @classmethod
    def get_instance(cls) -> OllamaLLMProvider:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_loaded_models(self) -> list[str]:
        """Query Ollama /api/ps for currently resident model names."""
        import urllib.request
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/ps", timeout=1.5) as resp:
                if resp.status != 200:
                    return []
                data = json.loads(resp.read().decode("utf-8"))
            return [str(m.get("name", "")) for m in data.get("models", []) if m.get("name")]
        except Exception:
            return []

    def evict_loaded_models(self, reason: str = "switch") -> list[str]:
        """
        Immediately unload any Ollama models held in daemon RAM via keep_alive=0
        so Transformers notes/summary generation does not contend for 8 GB system RAM.
        """
        import urllib.request
        loaded = self.get_loaded_models()
        if not loaded:
            return []
        before = get_process_memory_snapshot()
        evicted: list[str] = []
        for m_name in loaded:
            try:
                payload = {"model": m_name, "prompt": "", "keep_alive": 0, "stream": False}
                req = urllib.request.Request(
                    f"{self.base_url}/api/generate",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    resp.read()
                evicted.append(m_name)
            except Exception as e:
                logger.debug(f"Ollama eviction call failed for {m_name} ({reason}): {e}")
        # Confirm unload in /api/ps briefly
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if not self.get_loaded_models():
                break
            time.sleep(0.2)
        after = get_process_memory_snapshot()
        logger.info(
            f"[LLM_LIFECYCLE] action=evict_ollama_done reason={reason} evicted={evicted} "
            f"before={before} after={after}"
        )
        return evicted

    def is_available(self) -> bool:
        """Check whether the Ollama daemon is reachable and the configured model is installed."""
        import urllib.request
        import shutil
        import sys

        def _check_tags() -> bool:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2.0) as resp:
                if resp.status != 200:
                    return False
                data = json.loads(resp.read().decode("utf-8"))
            installed = {m.get("name", "") for m in data.get("models", [])}
            return (
                self.model_name in installed
                or f"{self.model_name}:latest" in installed
            )

        try:
            return _check_tags()
        except Exception as e:
            in_test = os.getenv("TESTING") == "1" or ("pytest" in sys.modules)
            if not in_test and ("127.0.0.1" in self.base_url or "localhost" in self.base_url):
                ollama_exe = shutil.which("ollama")
                if not ollama_exe:
                    candidate = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
                    if candidate.exists():
                        ollama_exe = str(candidate)
                if ollama_exe:
                    try:
                        env = os.environ.copy()
                        env.setdefault("OLLAMA_HOST", "127.0.0.1:11434")
                        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                        subprocess.Popen(
                            [ollama_exe, "serve"],
                            env=env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            creationflags=creationflags,
                        )
                        for _ in range(10):
                            time.sleep(0.5)
                            try:
                                if _check_tags():
                                    return True
                            except Exception:
                                pass
                    except Exception:
                        pass
            logger.debug(f"Ollama availability check failed: {e}")
            return False

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1,
        format_json: bool = False,
    ) -> str:
        """Generate text via Ollama /api/generate sequentially under lifecycle coordinator."""
        import urllib.request
        import urllib.error
        from app.config import OLLAMA_THINK

        effective_max_tokens = min(max(max_tokens, 900) if format_json else max(max_tokens, 512), 4096)
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "think": bool(OLLAMA_THINK),
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": max(0.0, float(temperature)),
                "num_predict": effective_max_tokens,
                "num_gpu": 99,
                "num_ctx": 4096,
                "repeat_penalty": 1.12,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        if format_json:
            payload["format"] = "json"

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        def _post() -> dict[str, Any]:
            t_inf0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
            logger.debug(
                f"[LLM_LIFECYCLE] action=infer_done provider=ollama model={self.model_name} "
                f"elapsed_ms={round((time.perf_counter() - t_inf0) * 1000, 1)}"
            )
            return res_data

        try:
            if _LIFECYCLE.current_thread_holds_lock():
                with self._gen_lock:
                    data = _post()
            else:
                with _LIFECYCLE.lease("ollama", feature="generate_text", release_local_on_exit=False):
                    with self._gen_lock:
                        data = _post()
        except urllib.error.URLError as e:
            raise LLMException(f"Ollama request failed ({self.base_url}): {e}") from e
        except Exception as e:
            raise LLMException(f"Ollama inference error: {e}") from e

        response = str(data.get("response", "")).strip()
        response = re.sub(r"<think>[\s\S]*?</think>", "", response, flags=re.IGNORECASE).strip()
        if not response and data.get("thinking"):
            thinking_txt = str(data.get("thinking", "")).strip()
            if format_json and "{" in thinking_txt:
                response = thinking_txt
        if not response:
            raise LLMException(f"Ollama returned an empty response for model '{self.model_name}'.")
        return LocalLLMProvider._detect_repetition(response)

    def stream_text(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 600,
        temperature: float = 0.25,
    ):
        """Stream response tokens live from Ollama /api/generate running on GPU."""
        import urllib.request
        import urllib.error
        from app.config import OLLAMA_THINK

        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "think": bool(OLLAMA_THINK),
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": max(0.0, float(temperature)),
                "num_predict": min(max(int(max_tokens), 128), 2048),
                "num_gpu": 99,
                "num_ctx": 4096,
                "repeat_penalty": 1.12,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        def _iter_stream():
            in_think = False
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except Exception:
                            continue
                        tok = chunk.get("response") or ""
                        if tok:
                            if "<think>" in tok:
                                in_think = True
                                tok = tok.split("<think>", 1)[0]
                            if "</think>" in tok:
                                in_think = False
                                tok = tok.split("</think>", 1)[1]
                            if not in_think and tok:
                                yield tok
                        if chunk.get("done"):
                            break
            except urllib.error.URLError as e:
                raise LLMException(f"Ollama streaming failed ({self.base_url}): {e}") from e

        if _LIFECYCLE.current_thread_holds_lock():
            with self._gen_lock:
                yield from _iter_stream()
        else:
            with _LIFECYCLE.lease("ollama", feature="stream_text", release_local_on_exit=False):
                with self._gen_lock:
                    yield from _iter_stream()

    def generate_structured(
        self,
        prompt: str,
        schema: Optional[dict[str, Any]] = None,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Generate structured JSON dictionary adhering to optional schema."""
        schema_instruction = ""
        if schema:
            schema_str = json.dumps(schema, indent=2)
            schema_instruction = (
                f"\nCRITICAL INSTRUCTION: You MUST output ONLY valid JSON adhering strictly to this schema:\n"
                f"{schema_str}\n"
                f"Do not write conversational preamble or postscript. Output ONLY raw JSON inside {{ ... }}."
            )
        full_system = f"{system_prompt}{schema_instruction}".strip()
        text = self.generate_text(
            prompt=prompt,
            system_prompt=full_system,
            max_tokens=max_tokens,
            temperature=temperature,
            format_json=True,
        )
        return LocalLLMProvider._extract_json(text)


def resolve_provider_name(feature: Optional[str] = None) -> str:
    """
    Resolve configured LLM provider for a feature without allowing global LLM_PROVIDER
    to silently override protected features (notes, summary, translation).
    """
    from app.config import (
        LLM_PROVIDER,
        NOTES_LLM_PROVIDER,
        SUMMARY_LLM_PROVIDER,
        TRANSLATION_LLM_PROVIDER,
        QUIZ_LLM_PROVIDER,
        RAG_LLM_PROVIDER,
    )
    f = (feature or "").strip().lower()
    if f in ("notes", "notes_summary"):
        return os.getenv("NOTES_LLM_PROVIDER", NOTES_LLM_PROVIDER).strip().lower()
    if f == "summary":
        return os.getenv("SUMMARY_LLM_PROVIDER", SUMMARY_LLM_PROVIDER).strip().lower()
    if f == "translation":
        return os.getenv("TRANSLATION_LLM_PROVIDER", TRANSLATION_LLM_PROVIDER).strip().lower()
    if f == "quiz":
        return os.getenv("QUIZ_LLM_PROVIDER", QUIZ_LLM_PROVIDER).strip().lower()
    if f == "rag":
        return os.getenv("RAG_LLM_PROVIDER", RAG_LLM_PROVIDER).strip().lower()
    return os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()


def get_llm_provider(feature: Optional[str] = None) -> BaseLLMProvider:
    """
    Factory function: returns active LLM provider for a given feature (or default config).
    Protected features ('notes', 'summary', 'translation') ignore global LLM_PROVIDER.
    No silent fallback from Ollama to Transformers unless ALLOW_OLLAMA_TO_LOCAL_FALLBACK is enabled.
    """
    from app.config import ENABLE_LOCAL_LLM, ALLOW_OLLAMA_TO_LOCAL_FALLBACK
    from roman_urdu_nlp.utils import env_flag
    import sys

    provider_name = resolve_provider_name(feature)
    in_test_mode = os.getenv("TESTING") == "1" or ("pytest" in sys.modules)
    if in_test_mode and not ENABLE_LOCAL_LLM:
        raise LLMException("No LLM provider available (ENABLE_LOCAL_LLM is disabled under test mode).")

    if provider_name == "ollama":
        ollama_p = OllamaLLMProvider.get_instance()
        if ollama_p.is_available():
            return ollama_p
        allow_fb = env_flag("ALLOW_OLLAMA_TO_LOCAL_FALLBACK", default=ALLOW_OLLAMA_TO_LOCAL_FALLBACK)
        if not allow_fb:
            logger.warning(
                f"Ollama LLM requested for feature='{feature or 'default'}' (model={ollama_p.model_name}) "
                "but unavailable; silent fallback to Transformers is disabled (ALLOW_OLLAMA_TO_LOCAL_FALLBACK=false)."
            )
            raise LLMException(
                f"Ollama provider unavailable for feature '{feature or 'default'}' "
                f"(model={ollama_p.model_name}) and fallback to Transformers is disabled."
            )
        logger.warning(
            f"Ollama LLM unavailable for feature='{feature or 'default'}'; "
            "explicitly falling back to LocalLLMProvider (ALLOW_OLLAMA_TO_LOCAL_FALLBACK=true)."
        )

    if provider_name == "cloud":
        cloud_p = CloudLLMProvider()
        if cloud_p.is_available():
            return cloud_p
        logger.warning("Cloud LLM requested but API key missing; falling back to local provider.")

    if ENABLE_LOCAL_LLM:
        return LocalLLMProvider.get_instance()

    raise LLMException("No LLM provider available (ENABLE_LOCAL_LLM is disabled).")
