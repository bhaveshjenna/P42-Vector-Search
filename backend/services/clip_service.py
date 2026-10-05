import torch
import numpy as np
from PIL import Image
from typing import List
from transformers import CLIPProcessor, CLIPModel
import logging

logger = logging.getLogger(__name__)


class CLIPService:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model '{self.model_id}' on {self.device}...")

        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model.eval()
        logger.info("CLIP model loaded successfully.")

    def _to_tensor(self, features) -> torch.Tensor:
        """
        Safely extract a raw tensor from model output.
        Transformers 5.x returns BaseModelOutputWithPooling instead of a plain tensor.
        We handle both cases here.
        """
        if isinstance(features, torch.Tensor):
            return features
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            return features.pooler_output
            
        raise ValueError(
            "Expected a raw torch.Tensor or a BaseModelOutputWithPooling (transformers 5.x), "
            f"but got {type(features)}. Unable to extract valid embeddings."
        )

    def _normalize(self, features) -> np.ndarray:
        """Extract, L2-normalize, and return as float32 numpy array of shape (N, 512)."""
        tensor = self._to_tensor(features)
        tensor = tensor / tensor.norm(p=2, dim=-1, keepdim=True)
        return tensor.cpu().numpy().astype(np.float32)

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Encode a single text string. Returns shape (1, 512)."""
        return self.get_text_embedding_batch([text])

    def get_text_embedding_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a list of text strings in one forward pass. Returns shape (N, 512)."""
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return self._normalize(features)

    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Encode a single PIL image. Returns shape (1, 512)."""
        return self.get_image_embedding_batch([image])

    def get_image_embedding_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Encode a list of PIL images in one forward pass. Returns shape (N, 512)."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return self._normalize(features)
