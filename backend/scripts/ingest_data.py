import os
import sys
import json
import logging
import csv
import argparse
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import settings
from services.clip_service import CLIPService
from services.faiss_service import FAISSService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Ingest Flickr30k images and captions into dual FAISS indexes.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images (for dry runs).")
    return parser.parse_args()


def parse_captions(captions_file: str) -> dict:
    """
    Parse the Flickr30k results.csv (pipe-delimited) into a dict:
        { "filename.jpg": ["caption1", "caption2", ...], ... }
    Drops malformed rows silently.
    """
    image_to_captions = {}
    if not os.path.exists(captions_file):
        logger.warning(f"Captions file not found at {captions_file}. Skipping caption ingestion.")
        return image_to_captions

    with open(captions_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="|")
        next(reader, None)  # skip header row

        for row in reader:
            if len(row) < 2:
                continue

            # Flickr30k format: image_name | comment_number | comment
            if len(row) >= 3 and row[1].strip().isdigit():
                img, cap = row[0].strip(), row[2].strip()
            else:
                img, cap = row[0].strip(), row[-1].strip()

            # Strip the #N caption index suffix if present (e.g. "12345.jpg#0" -> "12345.jpg")
            if "#" in img:
                img = img.split("#")[0]

            if not img or not cap:
                continue

            image_to_captions.setdefault(img, []).append(cap)

    total_caps = sum(len(v) for v in image_to_captions.values())
    logger.info(f"Parsed {total_caps} captions across {len(image_to_captions)} images.")
    return image_to_captions


def encode_captions_batch(clip_service: CLIPService, captions: list) -> np.ndarray:
    """Encode a list of caption strings in one batched forward pass."""
    if not captions:
        return np.empty((0, settings.EMBEDDING_DIM), dtype=np.float32)
    return clip_service.get_text_embedding_batch(captions)


def main():
    args = parse_arguments()

    if not os.path.exists(settings.IMAGES_DIR):
        logger.error(f"Images directory not found: {settings.IMAGES_DIR}")
        return

    logger.info("Initializing CLIP model and FAISS indexes...")
    clip_service = CLIPService(model_id=settings.MODEL_ID)
    image_faiss = FAISSService(embedding_dim=settings.EMBEDDING_DIM)
    caption_faiss = FAISSService(embedding_dim=settings.EMBEDDING_DIM)

    image_to_captions = parse_captions(settings.CAPTIONS_FILE)

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted([
        os.path.join(settings.IMAGES_DIR, f)
        for f in os.listdir(settings.IMAGES_DIR)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ])

    if not image_paths:
        logger.error("No images found in images directory.")
        return

    if args.limit:
        image_paths = image_paths[: args.limit]
        logger.info(f"DRY RUN: Limited to {args.limit} images.")

    logger.info(f"Starting ingestion of {len(image_paths)} images...")

    batch_size = 32
    image_mapping = {}
    caption_mapping = {}
    current_img_id = 0
    current_cap_id = 0
    skipped = 0

    total_batches = (len(image_paths) + batch_size - 1) // batch_size

    for batch_num, i in enumerate(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i : i + batch_size]

        batch_images = []
        batch_filenames = []
        batch_captions_per_image = []

        # Load and validate images in this batch
        for path in batch_paths:
            filename = os.path.basename(path)
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
                batch_filenames.append(filename)
                batch_captions_per_image.append(image_to_captions.get(filename, []))
            except Exception as e:
                logger.warning(f"Skipping {filename}: {e}")
                skipped += 1

        if not batch_images:
            continue

        # --- Encode all images in one batched call ---
        img_embeddings = clip_service.get_image_embedding_batch(batch_images)
        image_faiss.add_embeddings(img_embeddings)

        # --- Collect all captions for this batch, then encode in one batched call ---
        all_captions_flat = []
        caption_meta = []  # (img_id_in_current_batch, caption_text)

        for local_idx, captions in enumerate(batch_captions_per_image):
            img_id = current_img_id + local_idx
            image_mapping[str(img_id)] = {
                "filename": batch_filenames[local_idx],
                "captions": captions,
            }
            for cap in captions:
                if cap:
                    all_captions_flat.append(cap)
                    caption_meta.append((img_id, cap))

        if all_captions_flat:
            cap_embeddings = encode_captions_batch(clip_service, all_captions_flat)
            caption_faiss.add_embeddings(cap_embeddings)
            for (img_id, cap_text) in caption_meta:
                caption_mapping[str(current_cap_id)] = {
                    "image_id": img_id,
                    "text": cap_text,
                }
                current_cap_id += 1

        current_img_id += len(batch_images)
        logger.info(f"Batch {batch_num + 1}/{total_batches} — {current_img_id} images indexed.")

    logger.info(f"Ingestion complete. {current_img_id} images, {current_cap_id} captions. Skipped: {skipped}.")

    os.makedirs(settings.INDEX_DIR, exist_ok=True)
    image_faiss.save_index(settings.IMAGES_INDEX_FILE)
    caption_faiss.save_index(settings.CAPTIONS_INDEX_FILE)

    # indent=None writes compact JSON — significantly reduces file size for 30k images
    with open(settings.MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump({"images": image_mapping, "captions": caption_mapping}, f, indent=None)

    logger.info(f"Saved indexes and mapping to {settings.INDEX_DIR}")


if __name__ == "__main__":
    main()
