"""
RAG Knowledge Base API - Main Application Entry Point
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import logging

from app.api.routes import query, documents, health
from app.core.config import settings
from app.core.database import init_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    logger.info("🚀 Starting RAG Knowledge Base API...")
    await init_vector_store()
    logger.info("✅ Vector store initialized")
    yield
    logger.info("🛑 Shutting down RAG Knowledge Base API...")


app = FastAPI(
    title="RAG Knowledge Base API",
    description="""
    A production-ready Retrieval-Augmented Generation (RAG) API that combines
    document ingestion, semantic search, and LLM-powered answer generation.
    
    ## Features
    - 📄 Multi-format document ingestion (PDF, DOCX, TXT, CSV)
    - 🔍 Semantic search with FAISS vector store
    - 🤖 LLM-powered answers via OpenAI / Claude / Gemini
    - 📊 Source attribution and confidence scoring
    - 🔑 API key authentication
    - 📈 Usage analytics
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(query.router, prefix="/api/v1/query", tags=["Query"])


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
