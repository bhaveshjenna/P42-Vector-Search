from pydantic import BaseModel
from typing import List, Optional


class SearchTextRequest(BaseModel):
    query: str


class SearchResponseItem(BaseModel):
    faiss_id: int
    filename: str
    image_url: str
    similarity_pct: str
    latency_ms: float
    embedding_preview: str          # First 3 values of the 512-dim normalized embedding
    ground_truth_captions: List[str]
    matched_text: Optional[str] = None  # Only populated for image-to-text queries


class SearchResponse(BaseModel):
    results: List[SearchResponseItem]
    total_latency_ms: float
