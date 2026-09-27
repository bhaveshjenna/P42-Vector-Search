import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import logging

logger = logging.getLogger(__name__)

class CLIPService:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model {self.model_id} on {self.device}...")
        
        try:
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            self.model.eval()
            logger.info("CLIP model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise e

    def get_text_embedding(self, text: str) -> np.ndarray:
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            
            if not isinstance(text_features, torch.Tensor):
                text_features = text_features.pooler_output
            
            # Normalize the embeddings
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().numpy()
        except Exception as e:
            logger.error(f"Error extracting text embedding: {e}")
            raise e

    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            if not isinstance(image_features, torch.Tensor):
                image_features = image_features.pooler_output
            
            # Normalize the embeddings
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy()
        except Exception as e:
            logger.error(f"Error extracting image embedding: {e}")
            raise e
