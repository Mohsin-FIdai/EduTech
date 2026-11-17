"""
run_server.py — One-click launcher for the Pakistani English/Roman-Urdu Intelligence System.
Automatically finds an open port and launches the local web server.
"""

import sys
import os
import socket
import webbrowser
import threading
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import uvicorn

def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is free to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False

def find_available_port(preferred_ports=(8000, 8080, 8008, 5000)) -> int:
    """Find the first available port from the preferred list."""
    for p in preferred_ports:
        if is_port_available(p):
            return p
    return 8000

def open_browser_delayed(url: str, delay: float = 1.5):
    """Open browser automatically once server has started."""
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:
        pass

def ensure_ollama_running() -> None:
    """Ensure local Ollama daemon is running on 127.0.0.1:11434 for Qwen3.5 4B."""
    import urllib.request
    import shutil
    import subprocess
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.5):
            return
    except Exception:
        pass

    ollama_exe = shutil.which("ollama")
    if not ollama_exe:
        candidate = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
        if candidate.exists():
            ollama_exe = str(candidate)
    if not ollama_exe:
        return

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
                with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.5):
                    print("  [OK] Ollama daemon started (Qwen3.5 4B ready)")
                    return
            except Exception:
                pass
    except Exception:
        pass


if __name__ == "__main__":
    ensure_ollama_running()
    port = find_available_port()
    url = f"http://127.0.0.1:{port}"
    
    print("\n" + "=" * 70)
    print("  PAKISTANI ENGLISH / ROMAN-URDU LECTURE INTELLIGENCE SYSTEM")
    print("=" * 70)
    print(f"  --> Web Dashboard:   {url}")
    print(f"  --> Swagger API Docs: {url}/docs")
    print(f"  --> ReDoc:            {url}/redoc")
    print("=" * 70)
    print("  Opening your browser automatically...")
    print("  Keep this window open while using the application.")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 70 + "\n")

    # Launch browser automatically
    threading.Thread(target=open_browser_delayed, args=(url,), daemon=True).start()

    # Start FastAPI server
    try:
        uvicorn.run("app.main:app", host="127.0.0.1", port=port, reload=True, log_level="info")
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("  [INFO] Server stopped gracefully by user. (KeyboardInterrupt)")
        print("=" * 70 + "\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Server startup exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
