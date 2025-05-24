# app/models.py
from pydantic import BaseModel
from typing import Optional, List

class QueryRequest(BaseModel):
    """Model for query request"""
    question: str
    language: str = "vietnamese"
    enhance_retrieval: bool = True
    
class QueryResponse(BaseModel):
    """Model for query response"""
    answer: str
    question: str
    sources: Optional[List[dict]] = None