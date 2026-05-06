"""
Application Configuration using Pydantic Settings
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "RAG Knowledge Base API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ALLOWED_ORIGINS: List[str] = ["*"]

    # LLM Providers
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    DEFAULT_LLM_PROVIDER: str = "openai"  # openai | anthropic | gemini

    # LLM Settings
    LLM_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1024

    # Embedding
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536

    # Vector Store
    VECTOR_STORE_PATH: str = "./data/vector_store"
    VECTOR_STORE_TYPE: str = "faiss"  # faiss | chroma | pinecone

    # RAG Settings
    RETRIEVAL_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # Document Storage
    DOCUMENT_STORAGE_PATH: str = "./data/documents"
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt", ".csv", ".md"]

    # API Security
    API_KEY_HEADER: str = "X-API-Key"
    SECRET_KEY: str = "change-this-in-production"

    # Database (for metadata)
    DATABASE_URL: str = "sqlite:///./data/rag_metadata.db"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
