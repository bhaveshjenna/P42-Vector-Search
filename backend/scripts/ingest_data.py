import os
import sys
import json
import logging
import numpy as np
from PIL import Image

# Add the parent directory to sys.path to allow imports from core and services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import settings
from services.clip_service import CLIPService
from services.faiss_service import FAISSService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    if not os.path.exists(settings.IMAGES_DIR):
        logger.error(f"Images directory not found: {settings.IMAGES_DIR}")
        return

    logger.info("Initializing services...")
    clip_service = CLIPService(model_id=settings.MODEL_ID)
    faiss_service = FAISSService(embedding_dim=settings.EMBEDDING_DIM)

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = []
    
    for filename in os.listdir(settings.IMAGES_DIR):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            image_paths.append(os.path.join(settings.IMAGES_DIR, filename))
            
    if not image_paths:
        logger.info(f"No images found in {settings.IMAGES_DIR}")
        return

    logger.info(f"Found {len(image_paths)} images. Starting ingestion...")
    
    batch_size = 32
    id_mapping = {}
    current_id = 0
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_embeddings = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                embedding = clip_service.get_image_embedding(img)
                batch_embeddings.append(embedding[0])
                id_mapping[str(current_id)] = os.path.basename(path)
                current_id += 1
            except Exception as e:
                logger.error(f"Error processing image {path}: {e}")
                continue
                
        if batch_embeddings:
            embeddings_array = np.vstack(batch_embeddings)
            faiss_service.add_embeddings(embeddings_array)
            logger.info(f"Processed batch {i // batch_size + 1}/{(len(image_paths) + batch_size - 1) // batch_size}")

    # Save the index and mapping
    os.makedirs(settings.INDEX_DIR, exist_ok=True)
    faiss_service.save_index(settings.INDEX_FILE)
    
    try:
        with open(settings.MAPPING_FILE, 'w', encoding='utf-8') as f:
            json.dump(id_mapping, f, indent=4)
        logger.info(f"ID mapping saved to {settings.MAPPING_FILE}")
    except Exception as e:
        logger.error(f"Error saving ID mapping: {e}")

    logger.info("Ingestion completed successfully.")

if __name__ == "__main__":
    main()
