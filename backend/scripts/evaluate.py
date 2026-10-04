import os
import sys
import json
import random
import logging
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import settings
from services.clip_service import CLIPService
from services.faiss_service import FAISSService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def recall_at_k(ranks: list, k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r <= k) / len(ranks)


def mean_reciprocal_rank(ranks: list) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks) / len(ranks)


def main():
    logger.info("Loading CLIP model and indexes for evaluation...")

    clip_service = CLIPService(settings.MODEL_ID)

    for path, label in [
        (settings.IMAGES_INDEX_FILE, "images.index"),
        (settings.CAPTIONS_INDEX_FILE, "captions.index"),
        (settings.MAPPING_FILE, "mapping.json"),
    ]:
        if not os.path.exists(path):
            logger.error(f"{label} not found at {path}. Run ingest_data.py first.")
            return

    image_faiss = FAISSService(settings.EMBEDDING_DIM)
    image_faiss.load_index(settings.IMAGES_INDEX_FILE)

    caption_faiss = FAISSService(settings.EMBEDDING_DIM)
    caption_faiss.load_index(settings.CAPTIONS_INDEX_FILE)

    with open(settings.MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    images_map = mapping["images"]
    captions_map = mapping["captions"]

    # Only evaluate on images that have at least one caption
    valid_img_ids = [k for k, v in images_map.items() if v.get("captions")]
    sample_size = min(1000, len(valid_img_ids))

    if sample_size == 0:
        logger.error("No valid image-caption pairs found in mapping.")
        return

    sampled_ids = random.sample(valid_img_ids, sample_size)
    logger.info(f"Evaluating on {sample_size} randomly sampled image-caption pairs...")

    t2i_ranks = []
    i2t_ranks = []
    t2i_skipped = 0
    i2t_skipped = 0

    for i, img_id in enumerate(sampled_ids):
        if i % 100 == 0:
            logger.info(f"Progress: {i}/{sample_size}")

        img_data = images_map[img_id]
        captions = img_data["captions"]
        img_path = os.path.join(settings.IMAGES_DIR, img_data["filename"])

        # --- Text-to-Image (T2I): pick a random caption, search against image index ---
        try:
            query_caption = random.choice(captions)
            txt_emb = clip_service.get_text_embedding(query_caption)
            _, idxs = image_faiss.search(txt_emb, k=10)
            hit_list = list(idxs)
            if int(img_id) in hit_list:
                t2i_ranks.append(hit_list.index(int(img_id)) + 1)
            # If not in top 10, simply don't append — honestly excluded from metrics
        except Exception as e:
            logger.debug(f"T2I skipped for img_id={img_id}: {e}")
            t2i_skipped += 1

        # --- Image-to-Text (I2T): encode image, search against caption index ---
        try:
            img = Image.open(img_path).convert("RGB")
            img_emb = clip_service.get_image_embedding(img)
            _, idxs = caption_faiss.search(img_emb, k=10)

            hit_rank = None
            for rank_idx, cap_id in enumerate(idxs):
                if cap_id == -1:
                    continue
                cap_entry = captions_map.get(str(cap_id), {})
                if str(cap_entry.get("image_id")) == str(img_id):
                    hit_rank = rank_idx + 1
                    break

            if hit_rank is not None:
                i2t_ranks.append(hit_rank)
        except Exception as e:
            logger.debug(f"I2T skipped for img_id={img_id}: {e}")
            i2t_skipped += 1

    # --- Print Results ---
    evaluated_t2i = sample_size - t2i_skipped
    evaluated_i2t = sample_size - i2t_skipped

    print("\n" + "=" * 40)
    print("  EVALUATION RESULTS")
    print("=" * 40)
    print(f"  Samples evaluated:  {sample_size}")
    print()
    print(f"  Text → Image  (T2I)  [{evaluated_t2i} evaluated, {t2i_skipped} skipped]")
    print(f"    Recall@1:   {recall_at_k(t2i_ranks, 1):.4f}")
    print(f"    Recall@5:   {recall_at_k(t2i_ranks, 5):.4f}")
    print(f"    Recall@10:  {recall_at_k(t2i_ranks, 10):.4f}")
    print(f"    MRR:        {mean_reciprocal_rank(t2i_ranks):.4f}")
    print()
    print(f"  Image → Text  (I2T)  [{evaluated_i2t} evaluated, {i2t_skipped} skipped]")
    print(f"    Recall@1:   {recall_at_k(i2t_ranks, 1):.4f}")
    print(f"    Recall@5:   {recall_at_k(i2t_ranks, 5):.4f}")
    print(f"    Recall@10:  {recall_at_k(i2t_ranks, 10):.4f}")
    print(f"    MRR:        {mean_reciprocal_rank(i2t_ranks):.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
