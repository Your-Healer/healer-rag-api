"""Vector store initialization utilities."""
import re
import platform
from app.config import logger, vector_store, VECTOR_DB_TYPE, VectorDBType
from sqlalchemy import text


async def ensure_vector_store_initialized() -> bool:
    """
    Ensure the vector store is properly initialized.
    Make sure the vector store is properly initialized for the current platform.
    For Windows, we'll use synchronous mode.
    """
    try:
        # Check if vector store is AsyncPgVector
        if hasattr(vector_store, '_async_engine') and vector_store._async_engine:
            try:
                # Create the vector extension if needed
                async with vector_store._async_engine.begin() as conn:
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                return True
            except Exception as e:
                if "already exists" in str(e):
                    return True
                print(f"Extension creation warning (can be ignored): {str(e)}")
                return True

        # Simple test query to make sure the vector store is working
        _ = vector_store.similarity_search("test initialization", k=1)
        return True
    except Exception as e:
        error_str = str(e)

        # Log specific error info
        if "different vector dimensions" in error_str:
            dimensions = []
            dim_match = re.search(r'different vector dimensions (\d+) and (\d+)', error_str)
            if dim_match:
                dimensions = [dim_match.group(1), dim_match.group(2)]
                print(f"Vector dimension mismatch: {dimensions}")
                print("Please ensure your EMBEDDINGS_MODEL environment variable matches the model used to create the database")
        else:
            print(f"Vector store initialization error: {error_str}")

        return False
