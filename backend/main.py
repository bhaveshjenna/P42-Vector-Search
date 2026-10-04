import os
import json
import time
import logging
import urllib.parse
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve images as static files under /images
app.mount("/images", StaticFiles(directory=settings.IMAGES_DIR), name="images")


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
        similarity_pct=f"{max(0.0, float(dist)) * 100:.1f}%",
        latency_ms=round(latency_ms, 2),
        embedding_preview=preview,
        ground_truth_captions=img_data.get("captions", []),
        matched_text=matched_text,
    )


@app.post("/search/text", response_model=SearchResponse)
def search_text(request: SearchTextRequest):
    if image_faiss.index.ntotal == 0:
        raise HTTPException(status_code=503, detail="Image index is empty. Run ingest_data.py first.")
    try:
        t0 = time.perf_counter()
        embedding = clip_service.get_text_embedding(request.query)
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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image", response_model=SearchResponse)
def search_image(file: UploadFile = File(...)):
    if image_faiss.index.ntotal == 0:
        raise HTTPException(status_code=503, detail="Image index is empty. Run ingest_data.py first.")

    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")

    t0 = time.perf_counter()
    embedding = clip_service.get_image_embedding(image)
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


@app.post("/search/image-to-text", response_model=SearchResponse)
def search_image_to_text(file: UploadFile = File(...)):
    if caption_faiss.index.ntotal == 0:
        raise HTTPException(status_code=503, detail="Caption index is empty. Run ingest_data.py first.")

    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")

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
