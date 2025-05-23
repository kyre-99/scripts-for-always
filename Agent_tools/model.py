from pydantic import BaseModel
from typing import List, Optional
class RAGnInputModel(BaseModel):
    query: str
    
class VisualInputModel(BaseModel):
    image_url: str
    query:str = ""
    
class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False