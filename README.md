# 🧠 RAG Knowledge Base API

> A production-ready **Retrieval-Augmented Generation (RAG)** system built with FastAPI, FAISS, and multi-provider LLM support. Upload documents, query them in natural language, and get AI-generated answers grounded in your own data.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![LLM](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Claude%20%7C%20Gemini-purple)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

This API solves a critical problem in enterprise AI: **LLMs hallucinate when they don't know your data.** By combining semantic search (FAISS) with LLM generation (OpenAI / Anthropic / Gemini), this system answers questions **only from your uploaded documents**, with source attribution and confidence scores.

### Use Cases
- 📚 Internal knowledge base Q&A
- 🏥 Medical/legal document search
- 🛒 Product documentation chatbot
- 📊 Financial report analysis
- 🎓 Educational content assistant

---

## 🏗️ Architecture

```
Client Request
     │
     ▼
FastAPI Application
     │
     ├── POST /documents/upload   ──► Document Parser (PDF/DOCX/TXT/CSV)
     │                                       │
     │                                       ▼
     │                              Text Chunker (512 tokens, 64 overlap)
     │                                       │
     │                                       ▼
     │                              Embedding Client (OpenAI)
     │                                       │
     │                                       ▼
     │                              FAISS Vector Store (IndexFlatIP)
     │
     └── POST /query/             ──► Embed Query Vector
                                          │
                                          ▼
                                  FAISS Similarity Search (Top-K)
                                          │
                                          ▼
                                  Context Assembly
                                          │
                                          ▼
                                  LLM Generation (OpenAI/Anthropic/Gemini)
                                          │
                                          ▼
                                  Response + Source Attribution
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/rag-knowledge-base-api.git
cd rag-knowledge-base-api

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

Visit: **http://localhost:8000/docs** for the interactive Swagger UI

---

## 🐳 Docker Deployment

```bash
# Build and start
docker-compose up --build

# Stop
docker-compose down
```

---

## 📡 API Endpoints

### Upload a Document
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@your-document.pdf" \
  -F "collection=product-docs"
```

**Response:**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "your-document.pdf",
  "chunks_created": 42,
  "status": "success",
  "message": "Document ingested successfully into collection 'product-docs'"
}
```

### Query the Knowledge Base
```bash
curl -X POST "http://localhost:8000/api/v1/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the refund policies?",
    "top_k": 5,
    "collection": "product-docs"
  }'
```

**Response:**
```json
{
  "answer": "Based on Source 1, refunds are processed within 5-7 business days...",
  "sources": [
    {
      "doc_id": "abc123",
      "content": "Refunds are processed within 5-7 business days of approval...",
      "score": 0.934,
      "filename": "refund-policy.pdf",
      "collection": "product-docs"
    }
  ],
  "llm_provider": "openai",
  "model": "gpt-4o",
  "retrieval_time_ms": 12.4,
  "generation_time_ms": 834.2,
  "total_tokens": 487,
  "total_time_ms": 846.6
}
```

### Batch Query
```bash
curl -X POST "http://localhost:8000/api/v1/query/batch" \
  -H "Content-Type: application/json" \
  -d '["What is the return policy?", "How do I track my order?"]'
```

---

## 🔧 Configuration

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_LLM_PROVIDER` | `openai` | LLM backend: `openai`, `anthropic`, `gemini` |
| `LLM_MODEL` | `gpt-4o` | Model name |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `RETRIEVAL_TOP_K` | `5` | Chunks to retrieve per query |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `SIMILARITY_THRESHOLD` | `0.7` | Minimum cosine similarity score |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| LLM Providers | OpenAI GPT-4o, Anthropic Claude, Google Gemini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | FAISS (IndexFlatIP) |
| Document Parsing | pdfplumber, python-docx, pandas |
| Containerization | Docker + Docker Compose |
| Testing | pytest + pytest-asyncio |

---

## 🗂️ Project Structure

```
rag-knowledge-base-api/
├── app/
│   ├── main.py                 # FastAPI app entry point
│   ├── core/
│   │   ├── config.py           # Pydantic settings
│   │   ├── rag_engine.py       # Core RAG pipeline
│   │   └── database.py         # Storage initialization
│   ├── api/routes/
│   │   ├── documents.py        # Document ingestion endpoints
│   │   ├── query.py            # Query endpoints
│   │   └── health.py           # Health check
│   └── utils/
│       └── document_parser.py  # Multi-format file parser
├── tests/
│   └── test_api.py
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🔐 Security Notes

- API key authentication is scaffolded via `X-API-Key` header
- Never commit `.env` to version control
- Rotate `SECRET_KEY` in production
- Use HTTPS in production deployment

---

## 📈 Performance

| Operation | Typical Latency |
|---|---|
| Document ingestion (10-page PDF) | ~2-4 seconds |
| Query retrieval (FAISS) | < 20ms |
| LLM generation (GPT-4o) | 500-1500ms |
| End-to-end query | ~1-2 seconds |

---

## 🤝 Contributing

Pull requests welcome. Please open an issue first to discuss major changes.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by [Your Name](https://github.com/yourusername) — AI/ML Engineer specializing in LLM integration and RAG systems.*
