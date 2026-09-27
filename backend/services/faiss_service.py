import faiss
import numpy as np
import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)

class FAISSService:
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        # Using IndexFlatIP for Inner Product (Cosine Similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        logger.info(f"Initialized FAISS IndexFlatIP with dimension {self.embedding_dim}")

    def add_embeddings(self, embeddings: np.ndarray):
        if len(embeddings) == 0:
            return
        
        # FAISS requires float32
        embeddings = embeddings.astype(np.float32)
        try:
            self.index.add(embeddings)
            logger.info(f"Added {len(embeddings)} embeddings to the index. Total vectors: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Error adding embeddings to FAISS index: {e}")
            raise e

    def save_index(self, filepath: str):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            faiss.write_index(self.index, filepath)
            logger.info(f"FAISS index saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise e

    def load_index(self, filepath: str):
        if not os.path.exists(filepath):
            logger.error(f"Index file not found: {filepath}")
            raise FileNotFoundError(f"FAISS index file not found: {filepath}")
        
        try:
            self.index = faiss.read_index(filepath)
            logger.info(f"FAISS index loaded from {filepath}. Total vectors: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise e

    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self.index.ntotal == 0:
            logger.warning("Searching an empty index.")
            return np.array([]), np.array([])
        
        query_embedding = query_embedding.astype(np.float32)
        try:
            distances, indices = self.index.search(query_embedding, k)
            return distances[0], indices[0]
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}")
            raise e
