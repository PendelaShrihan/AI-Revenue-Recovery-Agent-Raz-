"""
Razorpay AI Revenue Recovery Agent - Main Application Entrypoint
FastAPI application with health checks, API routers, and static file serving.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB and connections if needed
    yield
    # Shutdown: Clean up resources

app = FastAPI(
    title="Razorpay AI Revenue Recovery Agent API",
    description="Autonomous diagnostic and revenue recovery engine for failed digital payments.",
    version="0.1.0",
    lifespan=lifespan
)

# Enable CORS for local development and dashboard UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "project": "AI Revenue Recovery Agent",
        "track": "Razorpay Hackathon Track 03",
        "status": "online",
        "version": "0.1.0"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "llm_provider": os.getenv("DEFAULT_LLM_PROVIDER", "mock")
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True)
