from pydantic import BaseModel
from typing import List

class SearchTextRequest(BaseModel):
    query: str

class SearchResponseItem(BaseModel):
    id: int
    path: str
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResponseItem]
