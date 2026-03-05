# api/main.py
"""
FastAPI backend for ClimSight.
Replaces the Streamlit UI layer with REST + WebSocket endpoints.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure src/climsight is on sys.path so engine imports work
_src_dir = os.path.join(os.path.dirname(__file__), os.pardir, "src", "climsight")
sys.path.insert(0, os.path.abspath(_src_dir))

from api.routes import sessions, analysis


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logging.info("🚀 ClimSight API starting up")
    os.makedirs("tmp/sandbox", exist_ok=True)
    yield
    logging.info("🛑 ClimSight API shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ClimSight API",
    version="0.1.0",
    description="Backend API for ClimSight – climate decision-support system.",
    lifespan=lifespan,
)

# CORS – allow the React dev server and any localhost origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",   # Vite default
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static files – serve generated plots & artifacts from sandbox
# ---------------------------------------------------------------------------
SANDBOX_ROOT = os.path.join(os.getcwd(), "tmp", "sandbox")
os.makedirs(SANDBOX_ROOT, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=SANDBOX_ROOT), name="artifacts")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
app.include_router(sessions.router, prefix="/api", tags=["Sessions"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "climsight-api"}
