import os
import json
import time
import logging
import urllib.parse
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np

from core.config import settings
from core.schemas import SearchTextRequest, SearchResponse, SearchResponseItem
from services.clip_service import CLIPService
from services.faiss_service import FAISSService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global state — populated during server startup
clip_service: CLIPService = None
image_faiss: FAISSService = None
caption_faiss: FAISSService = None
mapping: dict = {"images": {}, "captions": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global clip_service, image_faiss, caption_faiss, mapping

    logger.info("Server starting up — loading CLIP model and FAISS indexes...")

    clip_service = CLIPService(model_id=settings.MODEL_ID)

    image_faiss = FAISSService(embedding_dim=settings.EMBEDDING_DIM)
    if os.path.exists(settings.IMAGES_INDEX_FILE):
        image_faiss.load_index(settings.IMAGES_INDEX_FILE)
    else:
        logger.warning("images.index not found. Run ingest_data.py first.")

    caption_faiss = FAISSService(embedding_dim=settings.EMBEDDING_DIM)
    if os.path.exists(settings.CAPTIONS_INDEX_FILE):
        caption_faiss.load_index(settings.CAPTIONS_INDEX_FILE)
    else:
        logger.warning("captions.index not found. Run ingest_data.py first.")

    if os.path.exists(settings.MAPPING_FILE):
        with open(settings.MAPPING_FILE, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        logger.info(f"Mapping loaded: {len(mapping['images'])} images, {len(mapping['captions'])} captions.")
    else:
        logger.warning("mapping.json not found. Run ingest_data.py first.")

    yield
    logger.info("Server shutting down.")


app = FastAPI(title="Multimodal Image Search API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve images as static files under /images
os.makedirs(settings.IMAGES_DIR, exist_ok=True)
app.mount("/images", StaticFiles(directory=settings.IMAGES_DIR), name="images")

@app.get("/health")
def health(response: Response):
    m_loaded = clip_service is not None
    img_indexed = image_faiss.index.ntotal if image_faiss and hasattr(image_faiss, "index") and getattr(image_faiss.index, "ntotal", 0) > 0 else 0
    cap_indexed = caption_faiss.index.ntotal if caption_faiss and hasattr(caption_faiss, "index") and getattr(caption_faiss.index, "ntotal", 0) > 0 else 0
    map_loaded = bool(mapping and mapping.get("images") and mapping.get("captions"))
    
    is_healthy = m_loaded and img_indexed > 0 and cap_indexed > 0 and map_loaded
    
    if not is_healthy:
        response.status_code = 503
        
    return {
        "status": "ok" if is_healthy else "error",
        "model_loaded": m_loaded,
        "images_indexed": img_indexed,
        "captions_indexed": cap_indexed,
        "mapping_loaded": map_loaded,
    }



def build_result(idx: int, dist: float, img_data: dict, embedding: "np.ndarray", latency_ms: float, matched_text: str = None) -> SearchResponseItem:
    """Build a SearchResponseItem from FAISS result data."""
    filename = img_data.get("filename", "unknown.jpg")
    safe_filename = urllib.parse.quote(filename)
    # Use a relative URL so the app works on any host/port, not just localhost
    image_url = f"/images/{safe_filename}"
    # Show first 3 values of the embedding as a readable preview (L2 norm ≈ 1.0 sanity check)
    preview = f"{embedding[0]:.3f}, {embedding[1]:.3f}, {embedding[2]:.3f}"

    return SearchResponseItem(
        faiss_id=int(idx),
        filename=filename,
        image_url=image_url,
        similarity=f"{float(dist):.4f}",
        latency_ms=round(latency_ms, 2),
        embedding_preview=preview,
        ground_truth_captions=img_data.get("captions", []),
        matched_text=matched_text,
    )


@app.post("/search/text", response_model=SearchResponse)
def search_text(request: SearchTextRequest):
    q = request.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if image_faiss.index.ntotal == 0:
        raise HTTPException(status_code=503, detail="Image index is empty. Run ingest_data.py first.")
    try:
        t0 = time.perf_counter()
        embedding = clip_service.get_text_embedding(q)
        infer_ms = (time.perf_counter() - t0) * 1000

        distances, indices = image_faiss.search(embedding, k=5)
        total_ms = (time.perf_counter() - t0) * 1000

        results = []
        for dist, idx in zip(distances, indices):
            if idx == -1:
                continue
            img_data = mapping["images"].get(str(idx), {})
            results.append(build_result(idx, dist, img_data, embedding[0], infer_ms))

        return SearchResponse(results=results, total_latency_ms=round(total_ms, 2))
    except Exception as e:
        logger.exception(f"Error in /search/text: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/search/image", response_model=SearchResponse)
def search_image(file: UploadFile = File(...), exclude_near_duplicates: bool = False):
    if image_faiss.index.ntotal == 0:
        raise HTTPException(status_code=503, detail="Image index is empty. Run ingest_data.py first.")
    MAX_BYTES = 10 * 1024 * 1024
    file_bytes = file.file.read(MAX_BYTES + 1)
    if len(file_bytes) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10 MB.")
    
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        if image.width * image.height > 89478485:
            raise HTTPException(status_code=400, detail="Image dimensions are too large.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        t0 = time.perf_counter()
        embedding = clip_service.get_image_embedding(image)
        infer_ms = (time.perf_counter() - t0) * 1000

        distances, indices = image_faiss.search(embedding, k=6 if exclude_near_duplicates else 5)
        total_ms = (time.perf_counter() - t0) * 1000

        results = []
        for dist, idx in zip(distances, indices):
            if idx == -1:
                continue
            # NOTE: this only catches near-exact vector matches, not re-saved or resized copies
            if exclude_near_duplicates and float(dist) > 0.999:
                continue
                
            img_data = mapping["images"].get(str(idx), {})
            results.append(build_result(idx, dist, img_data, embedding[0], infer_ms))
            
            if len(results) == 5:
                break

        return SearchResponse(results=results, total_latency_ms=round(total_ms, 2))
    except Exception as e:
        logger.exception(f"Error in /search/image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/search/image-to-text", response_model=SearchResponse)
def search_image_to_text(file: UploadFile = File(...)):
    if caption_faiss.index.ntotal == 0:
        raise HTTPException(status_code=503, detail="Caption index is empty. Run ingest_data.py first.")
    MAX_BYTES = 10 * 1024 * 1024
    file_bytes = file.file.read(MAX_BYTES + 1)
    if len(file_bytes) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10 MB.")
    
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        if image.width * image.height > 89478485:
            raise HTTPException(status_code=400, detail="Image dimensions are too large.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        t0 = time.perf_counter()
        embedding = clip_service.get_image_embedding(image)
        infer_ms = (time.perf_counter() - t0) * 1000

        distances, indices = caption_faiss.search(embedding, k=5)
        total_ms = (time.perf_counter() - t0) * 1000

        results = []
        for dist, idx in zip(distances, indices):
            if idx == -1:
                continue
            cap_data = mapping["captions"].get(str(idx), {})
            img_id = str(cap_data.get("image_id", -1))
            img_data = mapping["images"].get(img_id, {})
            results.append(build_result(idx, dist, img_data, embedding[0], infer_ms, matched_text=cap_data.get("text")))

        return SearchResponse(results=results, total_latency_ms=round(total_ms, 2))
    except Exception as e:
        logger.exception(f"Error in /search/image-to-text: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
