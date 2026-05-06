"""
Query Routes - RAG-powered question answering endpoint
"""

import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.rag_engine import get_engine

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="Question to answer")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    collection: Optional[str] = Field(default=None, description="Filter by collection name")
    stream: Optional[bool] = Field(default=False, description="Stream response (future feature)")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the key features of the product?",
                "top_k": 5,
                "collection": "product-docs",
            }
        }


class SourceChunk(BaseModel):
    doc_id: str
    content: str
    score: float
    filename: str
    collection: Optional[str]


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    query: str
    llm_provider: str
    model: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_tokens: int
    total_time_ms: float


@router.post("/", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the RAG knowledge base with a natural language question.

    Returns an LLM-generated answer grounded in retrieved document chunks,
    along with source attribution and performance metrics.
    """
    engine = get_engine()

    try:
        result = await engine.query(request.question, top_k=request.top_k)
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

    sources = [
        SourceChunk(
            doc_id=s.doc_id,
            content=s.content[:500],  # truncate for response
            score=s.score,
            filename=s.metadata.get("filename", "unknown"),
            collection=s.metadata.get("collection"),
        )
        for s in result.sources
    ]

    return QueryResponse(
        answer=result.answer,
        sources=sources,
        query=result.query,
        llm_provider=result.llm_provider,
        model=result.model,
        retrieval_time_ms=round(result.retrieval_time_ms, 2),
        generation_time_ms=round(result.generation_time_ms, 2),
        total_tokens=result.total_tokens,
        total_time_ms=round(result.retrieval_time_ms + result.generation_time_ms, 2),
    )


@router.post("/batch")
async def batch_query(questions: List[str]):
    """Process multiple questions in a single request."""
    if len(questions) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 questions per batch request")

    engine = get_engine()
    results = []

    for question in questions:
        try:
            result = await engine.query(question)
            results.append({
                "question": question,
                "answer": result.answer,
                "sources_count": len(result.sources),
                "status": "success",
            })
        except Exception as e:
            results.append({
                "question": question,
                "answer": None,
                "status": "error",
                "error": str(e),
            })

    return {"results": results, "total": len(results)}
