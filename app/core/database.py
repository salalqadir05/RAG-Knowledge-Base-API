"""Vector store initialization."""
import logging
logger = logging.getLogger(__name__)

async def init_vector_store():
    """Initialize the vector store on startup."""
    import os
    os.makedirs("./data/vector_store", exist_ok=True)
    os.makedirs("./data/documents", exist_ok=True)
    logger.info("Storage directories ready.")
