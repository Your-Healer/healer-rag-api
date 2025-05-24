from langchain_core.embeddings import Embeddings

from .pg_vector_store import PGVectorStore

def get_vector_store(
    connection_string: str,
    embeddings: Embeddings,
    collection_name: str,
    async_mode: bool = False
):
    return PGVectorStore(
        connection_string=connection_string,
        embedding_function=embeddings,
        collection_name=collection_name,
        async_mode=async_mode
    )