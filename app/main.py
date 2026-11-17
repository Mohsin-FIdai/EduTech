"""
FastAPI Main Application entry point for the Pakistani English/Roman-Urdu Lecture & Meeting Intelligence System.
"""

import sys
import socket
# Set default global socket timeout to prevent indefinite network hangs in deep_translator / requests
socket.setdefaulttimeout(15.0)

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config import BASE_DIR, STORAGE_DIR, UPLOADS_DIR, PROCESSED_DIR
from app.database import engine, Base
from app.models import (
    Recording, Chunk, Transcript, TranscriptSegment, IntelligenceReport, SegmentEmbedding,
    Entity, EntityMention, StitchedBlock
)  # ensure models are registered
from app.services.intelligence_preprocessor import IntelligenceViewRecord  # register intelligence_views table
from app.routers import upload, recordings, text_processing, intelligence, search, stitching, export
from app.services.audio_validator import AudioValidationException


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event: create database tables and initialize storage directories."""
    # Ensure storage paths exist
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize SQLite database schema
    Base.metadata.create_all(bind=engine)

    # Pre-warm GPU embedding model and Ollama in background so first chat query starts streaming immediately
    import os
    import threading
    if "pytest" not in sys.modules and os.getenv("TESTING") != "1":
        def _warmup_models():
            try:
                from app.services.embedding_service import EmbeddingModelManager
                EmbeddingModelManager.get_model()
            except Exception:
                pass
            try:
                import json, urllib.request
                from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_KEEP_ALIVE
                req = urllib.request.Request(
                    f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
                    data=json.dumps({"model": OLLAMA_MODEL, "prompt": "", "keep_alive": OLLAMA_KEEP_ALIVE, "stream": False}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    resp.read()
            except Exception:
                pass
        threading.Thread(target=_warmup_models, daemon=True).start()

    yield


app = FastAPI(
    title="Pakistani English/Roman-Urdu Lecture & Meeting Intelligence API",
    description="Backend API for uploading, validating, chunking, and transcribing code-switched Pakistani lectures and meetings.",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# Global Exception Handlers ensuring structured JSON errors (No raw crashes)
@app.exception_handler(AudioValidationException)
async def audio_validation_exception_handler(request: Request, exc: AudioValidationException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "code": exc.code,
            "message": exc.message,
            "details": exc.details
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "code": "HTTP_ERROR",
            "message": str(exc.detail),
            "details": None
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "code": "INTERNAL_SERVER_ERROR",
            "message": f"An internal server error occurred: {str(exc)}",
            "details": None
        }
    )


# Include API Routers
app.include_router(upload.router)
app.include_router(recordings.router)
app.include_router(text_processing.router)
app.include_router(intelligence.router)
app.include_router(search.router)
app.include_router(stitching.router)
app.include_router(export.router)

# Mount Static Files
static_path = BASE_DIR / "app" / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the single-page frontend application."""
    index_file = BASE_DIR / "app" / "static" / "index.html"
    if index_file.exists():
        response = FileResponse(str(index_file))
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    return {"message": "Lecture & Meeting Intelligence API is running. Visit /docs for Swagger UI."}


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}
