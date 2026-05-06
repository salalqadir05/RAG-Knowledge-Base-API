"""
RAG Engine - Core Retrieval-Augmented Generation Logic
Handles document chunking, embedding, retrieval, and LLM answer generation.
"""

import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

import faiss
import numpy as np
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents an ingested document chunk."""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with similarity score."""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class RAGResponse:
    """Final RAG response with answer and sources."""
    answer: str
    sources: List[RetrievedChunk]
    query: str
    llm_provider: str
    model: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_tokens: int


class TextChunker:
    """Splits documents into overlapping chunks for embedding."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """Split text into overlapping chunks."""
        import uuid
        chunks = []
        words = text.split()
        step = self.chunk_size - self.chunk_overlap

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            if len(chunk_words) < 20:  # skip tiny tail chunks
                continue
            chunk_text = " ".join(chunk_words)
            chunks.append(
                Document(
                    doc_id=str(uuid.uuid4()),
                    content=chunk_text,
                    metadata={**metadata, "chunk_index": i // step},
                )
            )
        return chunks


class EmbeddingClient:
    """Generates embeddings using OpenAI text-embedding models."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.EMBEDDING_MODEL

    async def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts, returns (N, D) float32 array."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        vectors = [e.embedding for e in response.data]
        return np.array(vectors, dtype=np.float32)

    async def embed_one(self, text: str) -> np.ndarray:
        arr = await self.embed([text])
        return arr[0]


class VectorStore:
    """FAISS-backed vector store with metadata index."""

    def __init__(self, dimension: int = settings.EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after norm)
        self.documents: List[Document] = []

    def add(self, documents: List[Document]):
        """Add pre-embedded documents to the index."""
        vectors = np.stack([d.embedding for d in documents])
        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} chunks. Total: {len(self.documents)}")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[RetrievedChunk]:
        """Retrieve top-k most similar chunks."""
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < settings.SIMILARITY_THRESHOLD:
                continue
            doc = self.documents[idx]
            results.append(
                RetrievedChunk(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    score=float(score),
                    metadata=doc.metadata,
                )
            )
        return results

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}/index.faiss")
        logger.info(f"Vector store saved to {path}")

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}/index.faiss")
        logger.info(f"Vector store loaded from {path}")


class LLMClient:
    """Unified LLM client supporting OpenAI and Anthropic."""

    def __init__(self, provider: str = settings.DEFAULT_LLM_PROVIDER):
        self.provider = provider
        self.openai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.anthropic = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    async def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        """Generate an answer. Returns (answer_text, total_tokens)."""
        if self.provider == "openai":
            return await self._openai_generate(system_prompt, user_prompt)
        elif self.provider == "anthropic":
            return await self._anthropic_generate(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _openai_generate(self, system: str, user: str) -> tuple[str, int]:
        response = await self.openai.chat.completions.create(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens
        return answer, tokens

    async def _anthropic_generate(self, system: str, user: str) -> tuple[str, int]:
        response = await self.anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=settings.LLM_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        answer = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return answer, tokens


class RAGEngine:
    """
    Orchestrates the full RAG pipeline:
    Document → Chunk → Embed → Store → Retrieve → Generate
    """

    SYSTEM_PROMPT = """You are an expert assistant answering questions strictly based on the provided context.

Rules:
- Answer ONLY from the context below. Do not use external knowledge.
- If the context doesn't contain enough information, say so clearly.
- Be concise, factual, and cite which source (Source 1, 2...) your answer draws from.
- Format your response in clear, readable prose.

Context:
{context}
"""

    def __init__(self):
        self.chunker = TextChunker(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        self.embedder = EmbeddingClient()
        self.vector_store = VectorStore()
        self.llm = LLMClient()

    async def ingest(self, text: str, metadata: Dict[str, Any]) -> int:
        """Ingest a document: chunk → embed → store. Returns number of chunks."""
        chunks = self.chunker.split(text, metadata)
        texts = [c.content for c in chunks]
        embeddings = await self.embedder.embed(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        self.vector_store.add(chunks)
        return len(chunks)

    async def query(self, question: str, top_k: int = None) -> RAGResponse:
        """Full RAG query pipeline."""
        k = top_k or settings.RETRIEVAL_TOP_K

        # Retrieval
        t0 = time.perf_counter()
        q_emb = await self.embedder.embed_one(question)
        chunks = self.vector_store.search(q_emb, top_k=k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        if not chunks:
            return RAGResponse(
                answer="I couldn't find relevant information in the knowledge base for your question.",
                sources=[],
                query=question,
                llm_provider=self.llm.provider,
                model=settings.LLM_MODEL,
                retrieval_time_ms=retrieval_ms,
                generation_time_ms=0,
                total_tokens=0,
            )

        # Build context
        context_parts = [
            f"Source {i+1} (score={c.score:.3f}, file={c.metadata.get('filename','unknown')}):\n{c.content}"
            for i, c in enumerate(chunks)
        ]
        context = "\n\n---\n\n".join(context_parts)
        system_prompt = self.SYSTEM_PROMPT.format(context=context)

        # Generation
        t1 = time.perf_counter()
        answer, tokens = await self.llm.generate(system_prompt, question)
        generation_ms = (time.perf_counter() - t1) * 1000

        return RAGResponse(
            answer=answer,
            sources=chunks,
            query=question,
            llm_provider=self.llm.provider,
            model=settings.LLM_MODEL,
            retrieval_time_ms=retrieval_ms,
            generation_time_ms=generation_ms,
            total_tokens=tokens,
        )


# Singleton engine instance
_engine: Optional[RAGEngine] = None


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
