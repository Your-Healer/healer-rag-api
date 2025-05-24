from app.utils.langchain import create_llm, create_async_rag_chain, enhance_query, query_rag_system
from app.config import logger, vector_store

class LangChain:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = None
        self.qa_chain = None
        self.retriever = None
    
    async def initialize(self):
        """Async initialization method - call this after creating the instance"""
        try:
            self.llm = create_llm()
            # Pass the initialized vector_store that has async support
            self.qa_chain, self.retriever = await create_async_rag_chain()
            return True
        except Exception as e:
            print(f"Failed to initialize LangChain: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    async def query(self, question: str, language: str = "vietnamese"):
        """Process a query through the RAG chain"""
        if not self.qa_chain:
            success = await self.initialize()
            if not success:
                raise ValueError("Failed to initialize LangChain")
            
        # enhanced_question = enhance_query(question, language).lower()
        enhanced_question = question.lower()
        response = await query_rag_system(self.qa_chain, enhanced_question, self.retriever, language)
        return response
    
def create_langchain(vector_store):
    """
    Create a LangChain instance with the provided vector store.
    Note: You must call langchain.initialize() after creating it.
    """
    return LangChain(vector_store)