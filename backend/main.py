import os
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from core.config import settings
from core.schemas import SearchTextRequest, SearchResponse, SearchResponseItem
from services.clip_service import CLIPService
from services.faiss_service import FAISSService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global instances
clip_service = None
faiss_service = None
id_mapping = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global clip_service, faiss_service, id_mapping
    
    logger.info("Starting up FastAPI server. Loading models and index...")
    
    try:
        clip_service = CLIPService(model_id=settings.MODEL_ID)
    except Exception as e:
        logger.error(f"Failed to load CLIP service: {e}")
        raise RuntimeError(f"Could not load CLIP model: {e}")
        
    try:
        faiss_service = FAISSService(embedding_dim=settings.EMBEDDING_DIM)
        faiss_service.load_index(settings.INDEX_FILE)
    except Exception as e:
        logger.error(f"Failed to load FAISS index. You might need to run the ingestion script first. Error: {e}")
        # Not throwing here so the app can still start, but searches will fail if no index
        
    try:
        if os.path.exists(settings.MAPPING_FILE):
            with open(settings.MAPPING_FILE, 'r', encoding='utf-8') as f:
                id_mapping = json.load(f)
            logger.info(f"Loaded ID mapping with {len(id_mapping)} entries.")
        else:
            logger.warning(f"ID mapping file not found at {settings.MAPPING_FILE}.")
    except Exception as e:
        logger.error(f"Error loading ID mapping: {e}")
        
    yield
    
    logger.info("Shutting down FastAPI server.")

app = FastAPI(title="P42 Multimodal Vector Search", lifespan=lifespan)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=settings.IMAGES_DIR), name="images")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "P42 Search Engine is running"}

@app.post("/search/text", response_model=SearchResponse)
def search_text(request: SearchTextRequest):
    if not clip_service or not faiss_service:
        raise HTTPException(status_code=500, detail="Services not fully initialized.")
        
    if faiss_service.index.ntotal == 0:
        raise HTTPException(status_code=400, detail="Search index is empty.")
        
    try:
        embedding = clip_service.get_text_embedding(request.query)
        distances, indices = faiss_service.search(embedding, k=5)
        
        results = []
        for dist, idx in zip(distances, indices):
            if idx == -1:
                continue
            str_idx = str(idx)
            import urllib.parse
            raw_path = id_mapping.get(str_idx, "unknown.jpg")
            filename = raw_path.replace('\\', '/').split('/')[-1]
            safe_filename = urllib.parse.quote(filename)
            url_path = f"http://localhost:8000/images/{safe_filename}"
            results.append(SearchResponseItem(id=int(idx), path=url_path, score=float(dist)))
            
        return SearchResponse(results=results)
    except Exception as e:
        logger.error(f"Error during text search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/image", response_model=SearchResponse)
def search_image(file: UploadFile = File(...)):
    if not clip_service or not faiss_service:
        raise HTTPException(status_code=500, detail="Services not fully initialized.")
        
    if faiss_service.index.ntotal == 0:
        raise HTTPException(status_code=400, detail="Search index is empty.")
        
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        embedding = clip_service.get_image_embedding(image)
        distances, indices = faiss_service.search(embedding, k=5)
        
        results = []
        for dist, idx in zip(distances, indices):
            if idx == -1:
                continue
            str_idx = str(idx)
            import urllib.parse
            raw_path = id_mapping.get(str_idx, "unknown.jpg")
            filename = raw_path.replace('\\', '/').split('/')[-1]
            safe_filename = urllib.parse.quote(filename)
            url_path = f"http://localhost:8000/images/{safe_filename}"
            results.append(SearchResponseItem(id=int(idx), path=url_path, score=float(dist)))
            
        return SearchResponse(results=results)
    except Exception as e:
        logger.error(f"Error during image search: {e}")
        raise HTTPException(status_code=500, detail=str(e))
