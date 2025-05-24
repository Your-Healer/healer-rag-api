import os
from typing import List, Dict, Any, Tuple, Optional
import psycopg
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_core.runnables.config import run_in_executor

def check_pgvector_connection(connection_string=None):
    """
    Check if the PostgreSQL connection is healthy and pgvector extension is installed.
    
    Args:
        connection_string: PostgreSQL connection string
        
    Returns:
        Tuple[bool, str]: (is_healthy, message)
    """
    # Use environment variable if connection string not provided
    connection_string = connection_string or os.environ.get("PGVECTOR_CONNECTION_STRING")
    if not connection_string:
        conn_params = {
            "host": os.environ.get("PGVECTOR_HOST", "localhost"),
            "port": os.environ.get("PGVECTOR_PORT", "5432"),
            "database": os.environ.get("PGVECTOR_DATABASE", "healer_vector_db"),
            "user": os.environ.get("PGVECTOR_USER", "postgres"),
            "password": os.environ.get("PGVECTOR_PASSWORD", ""),
        }
        connection_string = f"postgresql://{conn_params['user']}:{conn_params['password']}@{conn_params['host']}:{conn_params['port']}/{conn_params['database']}"
    
    try:
        # Parse connection string to get individual parameters
        # This is a simple parser, might need improvement for complex connection strings
        conn_parts = connection_string.replace('postgresql://', '').split('@')
        user_pass = conn_parts[0].split(':')
        host_port_db = conn_parts[1].split('/')
        
        user = user_pass[0]
        password = user_pass[1] if len(user_pass) > 1 else ""
        
        host_port = host_port_db[0].split(':')
        host = host_port[0]
        port = host_port[1] if len(host_port) > 1 else "5432"
        
        database = host_port_db[1]
        
        # Connect to the database
        conn = psycopg.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Check if pgvector extension is installed
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
        pgvector_installed = cursor.fetchone() is not None
        
        # Check PostgreSQL version
        cursor.execute("SELECT version();")
        pg_version = cursor.fetchone()[0]
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        if pgvector_installed:
            return True, f"Connection successful. PostgreSQL: {pg_version}. pgvector extension is installed."
        else:
            return False, f"Connection successful, but pgvector extension is not installed. PostgreSQL: {pg_version}"
    
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

class PGVectorStore(PGVector):
    """PostgreSQL-based vector store implementation using pgvector extension."""
    
    def __init__(self, connection_string=None, embedding_function=None, collection_name="healer_medical_docs", async_mode: bool = False):
        """
        Initialize the PGVector store.
        
        Args:
            connection_string: PostgreSQL connection string
            embedding_function: Embedding function to use
            collection_name: Name of the collection
        """
        # Use environment variable if connection string not provided
        self.connection_string = connection_string or os.environ.get("PGVECTOR_CONNECTION_STRING")
        if not self.connection_string:
            raise ValueError("PostgreSQL connection string must be provided or set in PGVECTOR_CONNECTION_STRING environment variable")
        
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.async_mode = async_mode
        
        # Initialize the PGVector store
        self._initialize_store()
        
    def _initialize_store(self):
        self.store = PGVector(
            collection_name=self.collection_name,
            # connection_url=self.connection_string,  # Changed from connection_string to connection_url
            connection=self.connection_string,
            embeddings=self.embedding_function,
            use_jsonb=True,
            async_mode=self.async_mode
        )
    
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of texts to add
            metadatas: List of metadata dictionaries
            
        Returns:
            List of IDs of the added texts
        """
        return self.store.add_texts(texts=texts, metadatas=metadatas)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of documents to return
            
        Returns:
            List of similar documents
        """
        return self.store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple[Document, float]]:
        """
        Search for similar documents with scores.
        
        Args:
            query: Query text
            k: Number of documents to return
            
        Returns:
            List of tuples of document and score
        """
        return self.store.similarity_search_with_score(query, k=k)
    
    def similarity_search_with_filter(self, query: str, filter: Dict[str, Any], k: int = 4) -> List[Document]:
        """
        Search for similar documents with filter.
        
        Args:
            query: Query text
            filter: Filter to apply
            k: Number of documents to return
            
        Returns:
            List of documents
        """
        # Convert filter to a format compatible with PGVector
        filter_string = self._format_filter(filter)
        docs = self.store.similarity_search(
            query=query,
            k=k,
            where=filter_string
        )
        return docs
    
    def _format_filter(self, filter: Dict[str, Any]) -> str:
        """
        Format filter dictionary to SQL WHERE clause format.
        
        Args:
            filter: Filter dictionary
            
        Returns:
            str: SQL WHERE clause
        """
        conditions = []
        for key, value in filter.items():
            if isinstance(value, str):
                conditions.append(f"metadata->'{key}' = '{value}'")
            elif isinstance(value, (int, float, bool)):
                conditions.append(f"metadata->'{key}' = '{value}'")
            elif value is None:
                conditions.append(f"metadata->'{key}' IS NULL")
        
        return " AND ".join(conditions) if conditions else ""
    
    def as_retriever(self, search_type="similarity", search_kwargs=None):
        """
        Create a retriever from the vector store.
        
        Args:
            search_type: Type of search to use
            search_kwargs: Search kwargs
            
        Returns:
            Retriever object
        """
        return self.store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def delete_collection(self):
        self.store.delete_collection()
    
    def check_health(self) -> Tuple[bool, str]:
        """
        Check if the PGVector store is healthy.
        
        Returns:
            Tuple[bool, str]: (is_healthy, message)
        """
        try:
            # Check basic connection
            connection_status, message = check_pgvector_connection(self.connection_string)
            if not connection_status:
                return False, message
            
            # Check if collection exists by attempting a simple query
            try:
                _ = self.store.similarity_search("health check", k=1)
                return True, f"PGVector store is healthy. Collection: {self.collection_name}"
            except Exception as e:
                if "relation" in str(e) and "does not exist" in str(e):
                    return False, f"Collection '{self.collection_name}' does not exist. Database connection is healthy."
                else:
                    return False, f"Collection check failed: {str(e)}"
                
        except Exception as e:
            return False, f"Health check failed: {str(e)}"
    