"""
Document Ingestion Routes
Handles upload, parsing, and indexing of various file types.
"""

import os
import uuid
import logging
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form
from pydantic import BaseModel

from app.core.config import settings
from app.core.rag_engine import get_engine
from app.utils.document_parser import parse_document

logger = logging.getLogger(__name__)
router = APIRouter()


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    status: str
    message: str


class DocumentListResponse(BaseModel):
    documents: List[dict]
    total: int


@router.post("/upload", response_model=IngestResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection: Optional[str] = Form(default="default"),
    description: Optional[str] = Form(default=""),
):
    """
    Upload and ingest a document into the RAG knowledge base.
    
    Supported formats: PDF, DOCX, TXT, CSV, MD
    """
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Max allowed: {settings.MAX_FILE_SIZE_MB}MB",
        )

    # Save file
    doc_id = str(uuid.uuid4())
    save_path = Path(settings.DOCUMENT_STORAGE_PATH) / f"{doc_id}{ext}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(content)

    # Parse document
    try:
        text = parse_document(save_path, ext)
    except Exception as e:
        logger.error(f"Failed to parse {file.filename}: {e}")
        raise HTTPException(status_code=422, detail=f"Could not parse document: {str(e)}")

    # Ingest into RAG engine
    metadata = {
        "document_id": doc_id,
        "filename": file.filename,
        "collection": collection,
        "description": description,
        "file_size_mb": round(size_mb, 3),
    }

    engine = get_engine()
    chunks = await engine.ingest(text, metadata)

    logger.info(f"Ingested '{file.filename}' → {chunks} chunks (id={doc_id})")

    return IngestResponse(
        document_id=doc_id,
        filename=file.filename,
        chunks_created=chunks,
        status="success",
        message=f"Document ingested successfully into collection '{collection}'",
    )


@router.post("/ingest-text", response_model=IngestResponse)
async def ingest_text(
    text: str = Form(...),
    title: str = Form(...),
    collection: Optional[str] = Form(default="default"),
):
    """Ingest raw text directly into the knowledge base."""
    if len(text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Text too short (minimum 50 characters)")

    doc_id = str(uuid.uuid4())
    metadata = {
        "document_id": doc_id,
        "filename": title,
        "collection": collection,
        "source": "direct_text",
    }

    engine = get_engine()
    chunks = await engine.ingest(text, metadata)

    return IngestResponse(
        document_id=doc_id,
        filename=title,
        chunks_created=chunks,
        status="success",
        message=f"Text ingested successfully into collection '{collection}'",
    )
