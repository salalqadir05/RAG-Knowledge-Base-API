"""
Integration tests for RAG Knowledge Base API
"""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock
import numpy as np

from app.main import app


@pytest.fixture
def mock_engine():
    """Mock the RAG engine for testing without real LLM calls."""
    from app.core.rag_engine import RAGResponse, RetrievedChunk
    engine = MagicMock()
    engine.ingest = AsyncMock(return_value=5)
    engine.query = AsyncMock(return_value=RAGResponse(
        answer="The product supports PDF, DOCX, and TXT formats.",
        sources=[
            RetrievedChunk(
                doc_id="test-id-123",
                content="Our product supports multiple file formats including PDF, DOCX, and TXT.",
                score=0.92,
                metadata={"filename": "product-docs.pdf", "collection": "default"},
            )
        ],
        query="What file formats are supported?",
        llm_provider="openai",
        model="gpt-4o",
        retrieval_time_ms=45.2,
        generation_time_ms=820.5,
        total_tokens=312,
    ))
    return engine


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


@pytest.mark.asyncio
async def test_query_endpoint(mock_engine):
    with patch("app.api.routes.query.get_engine", return_value=mock_engine):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query/",
                json={"question": "What file formats are supported?", "top_k": 3},
            )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["sources"]) == 1
    assert data["sources"][0]["score"] == 0.92
    assert data["total_tokens"] == 312


@pytest.mark.asyncio
async def test_query_too_short():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/query/", json={"question": "Hi"})
    assert response.status_code == 422  # Validation error - too short


@pytest.mark.asyncio
async def test_batch_query_limit():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query/batch",
            json=["Q" * 5] * 11,  # 11 questions - exceeds limit
        )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_unsupported_type():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.exe", b"binary content", "application/octet-stream")},
        )
    assert response.status_code == 415
