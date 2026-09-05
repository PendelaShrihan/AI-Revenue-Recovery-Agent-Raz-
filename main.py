"""
Razorpay AI Revenue Recovery Agent - Main Application Entrypoint
FastAPI application with health checks, API routers, and static file serving.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Load environment variables
load_dotenv()

# ── Rate limiter: 100 requests / minute per client IP ─────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB and connections if needed
    yield
    # Shutdown: Clean up resources


app = FastAPI(
    title="Razorpay AI Revenue Recovery Agent API",
    description="Autonomous diagnostic and revenue recovery engine for failed digital payments.",
    version="0.1.0",
    lifespan=lifespan,
)

# Attach limiter state and exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

from api_integration import webhook_router
from api_integration.rest_router import rest_router

# Enable CORS for local development and dashboard UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(webhook_router)
app.include_router(rest_router)
app.include_router(rest_router, prefix="/api")


@app.get("/")
def root():
    return {
        "project": "AI Revenue Recovery Agent",
        "engine": "Autonomous Revenue Recovery Engine",
        "status": "online",
        "version": "0.1.0",
        "dashboard": "/dashboard/index.html",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "llm_provider": os.getenv("DEFAULT_LLM_PROVIDER", "mock"),
    }


# Serve merchant dashboard — available at /dashboard/index.html
import pathlib
_frontend_dir = pathlib.Path(__file__).parent / "frontend"
if _frontend_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(_frontend_dir), html=True), name="dashboard")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True)
