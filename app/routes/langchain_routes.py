# app/routes/query_routes.py
import traceback
from fastapi import APIRouter, HTTPException, Depends

from app.config import vector_store
from app.services.langchain import create_langchain
from app.models import QueryRequest, QueryResponse

router = APIRouter()

# Create the instance but don't initialize it yet
langchain = create_langchain(vector_store=vector_store)

# Dependency to ensure langchain is initialized
async def get_initialized_langchain():
    if langchain.qa_chain is None:
        await langchain.initialize()
    return langchain

@router.post("/langchain/query", response_model=QueryResponse)
async def query(request: QueryRequest, lc=Depends(get_initialized_langchain)):
    """Process a query for the RAG system."""
    try:
        # Process the query using the initialized langchain
        result = await lc.query(request.question, request.language)
        
        # Extract answer and sources from result
        answer = result.get("result", "")
        
        # Format sources if available
        sources = None
        if "retrieved_documents" in result and result["retrieved_documents"]:
            sources = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["retrieved_documents"]
            ]
        
        response: QueryResponse = {
            "answer": answer,
            "question": request.question,
            "sources": sources
        } 

        return response
    except Exception as e:
        print(
            f"Error processing query | Question: {request.question} | Error: {str(e)} | Traceback: {traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")