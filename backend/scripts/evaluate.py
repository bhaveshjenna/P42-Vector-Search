import os
import sys
import json
import random
import logging
import argparse
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


def evaluate_seed(seed: int, sample_size: int, gallery_size: int, valid_img_ids: list, images_map: dict, captions_map: dict, clip_service, all_image_vectors, all_caption_vectors):
    import faiss
    random.seed(seed)
    
    actual_gallery_size = min(gallery_size, len(valid_img_ids)) if gallery_size else len(valid_img_ids)
    gallery_subset = set(random.sample(valid_img_ids, actual_gallery_size))
    
    actual_sample_size = min(sample_size, len(gallery_subset))
    sampled_ids = random.sample(list(gallery_subset), actual_sample_size)
    
    logger.info(f"[Seed {seed}] Evaluating on {actual_sample_size} queries against a gallery of {actual_gallery_size} images...")

    # Build sub-indexes if we have a gallery_size, else use full vectors (but we need faiss index either way)
    # Actually, we can just always use IndexIDMap for simplicity
    gallery_img_ids = [int(i) for i in gallery_subset]
    sub_image_faiss = faiss.IndexIDMap(faiss.IndexFlatIP(settings.EMBEDDING_DIM))
    sub_image_faiss.add_with_ids(all_image_vectors[gallery_img_ids], np.array(gallery_img_ids, dtype=np.int64))

    gallery_cap_ids = []
    for cap_id_str, cap_entry in captions_map.items():
        if str(cap_entry.get("image_id")) in gallery_subset:
            gallery_cap_ids.append(int(cap_id_str))
            
    sub_caption_faiss = faiss.IndexIDMap(faiss.IndexFlatIP(settings.EMBEDDING_DIM))
    sub_caption_faiss.add_with_ids(all_caption_vectors[gallery_cap_ids], np.array(gallery_cap_ids, dtype=np.int64))

    t2i_ranks = []
    i2t_ranks = []
    t2i_skipped = 0
    i2t_skipped = 0
    t2i_misses = 0
    i2t_misses = 0
    
    for i, img_id in enumerate(sampled_ids):
        if i > 0 and i % 100 == 0:
            logger.info(f"[Seed {seed}] Progress: {i}/{actual_sample_size}")

        img_data = images_map[img_id]
        captions = img_data["captions"]
        img_path = os.path.join(settings.IMAGES_DIR, img_data["filename"])

        # --- Text-to-Image (T2I) ---
        try:
            query_caption = random.choice(captions)
            txt_emb = clip_service.get_text_embedding(query_caption)
            _, idxs = sub_image_faiss.search(txt_emb, k=10)
            
            hit_list = list(idxs[0])
            if int(img_id) in hit_list:
                t2i_ranks.append(hit_list.index(int(img_id)) + 1)
            else:
                t2i_ranks.append(float('inf'))
                t2i_misses += 1
        except Exception as e:
            logger.debug(f"[Seed {seed}] T2I skipped for img_id={img_id}: {e}")
            t2i_skipped += 1

        # --- Image-to-Text (I2T) ---
        try:
            img = Image.open(img_path).convert("RGB")
            img_emb = clip_service.get_image_embedding(img)
            _, idxs = sub_caption_faiss.search(img_emb, k=10)

            hit_rank = None
            for rank_idx, cap_id in enumerate(idxs[0]):
                if cap_id == -1:
                    continue
                
                cap_entry = captions_map.get(str(cap_id), {})
                if str(cap_entry.get("image_id")) == str(img_id):
                    hit_rank = rank_idx + 1
                    break

            if hit_rank is not None:
                i2t_ranks.append(hit_rank)
            else:
                i2t_ranks.append(float('inf'))
                i2t_misses += 1
        except Exception as e:
            logger.debug(f"[Seed {seed}] I2T skipped for img_id={img_id}: {e}")
            i2t_skipped += 1

    return {
        "t2i_evaluated": actual_sample_size - t2i_skipped,
        "i2t_evaluated": actual_sample_size - i2t_skipped,
        "t2i_skipped": t2i_skipped,
        "i2t_skipped": i2t_skipped,
        "t2i_misses": t2i_misses,
        "i2t_misses": i2t_misses,
        "t2i_r1": recall_at_k(t2i_ranks, 1),
        "t2i_r5": recall_at_k(t2i_ranks, 5),
        "t2i_r10": recall_at_k(t2i_ranks, 10),
        "t2i_mrr": mean_reciprocal_rank(t2i_ranks),
        "i2t_r1": recall_at_k(i2t_ranks, 1),
        "i2t_r5": recall_at_k(i2t_ranks, 5),
        "i2t_r10": recall_at_k(i2t_ranks, 10),
        "i2t_mrr": mean_reciprocal_rank(i2t_ranks),
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate multimodal search on Flickr30k")
    parser.add_argument("--seed", type=int, default=42, help="Starting random seed for reproducibility (default: 42)")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of queries to sample (default: 1000)")
    parser.add_argument("--gallery-size", type=int, default=None, help="Restrict the search gallery to N images")
    parser.add_argument("--num-seeds", type=int, default=1, help="Number of seeds to average over")
    args = parser.parse_args()

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

    valid_img_ids = [k for k, v in images_map.items() if v.get("captions")]
    
    if len(valid_img_ids) == 0:
        logger.error("No valid image-caption pairs found in mapping.")
        return

    # Extract all vectors so we can easily slice subsets
    all_image_vectors = image_faiss.index.reconstruct_n(0, image_faiss.index.ntotal)
    all_caption_vectors = caption_faiss.index.reconstruct_n(0, caption_faiss.index.ntotal)

    all_results = []
    
    for i in range(args.num_seeds):
        current_seed = args.seed + i
        res = evaluate_seed(
            current_seed, 
            args.sample_size, 
            args.gallery_size, 
            valid_img_ids, 
            images_map, 
            captions_map, 
            clip_service, 
            all_image_vectors, 
            all_caption_vectors
        )
        all_results.append(res)
        
    # Aggregate results
    keys = [
        "t2i_r1", "t2i_r5", "t2i_r10", "t2i_mrr", 
        "i2t_r1", "i2t_r5", "i2t_r10", "i2t_mrr"
    ]
    
    avg_results = {k: np.mean([r[k] for r in all_results]) for k in keys}
    std_results = {k: np.std([r[k] for r in all_results]) for k in keys}
    
    # Just grab the last run's counts (they should be identical if sample size doesn't change)
    eval_t2i = all_results[-1]["t2i_evaluated"]
    eval_i2t = all_results[-1]["i2t_evaluated"]
    skip_t2i = all_results[-1]["t2i_skipped"]
    skip_i2t = all_results[-1]["i2t_skipped"]
    miss_t2i = all_results[-1]["t2i_misses"]
    miss_i2t = all_results[-1]["i2t_misses"]
    
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Queries evaluated per seed: {eval_t2i}")
    print(f"  Gallery size:               {args.gallery_size if args.gallery_size else len(valid_img_ids)}")
    print(f"  Averaged over {args.num_seeds} seeds (starting from {args.seed})")
    print()
    print(f"  Text -> Image  (T2I)  [{eval_t2i} evaluated, {skip_t2i} skipped]")
    print(f"    Missed Top 10: {miss_t2i} (last run)")
    print(f"    Recall@1:      {avg_results['t2i_r1']:.4f} ± {std_results['t2i_r1']:.4f}")
    print(f"    Recall@5:      {avg_results['t2i_r5']:.4f} ± {std_results['t2i_r5']:.4f}")
    print(f"    Recall@10:     {avg_results['t2i_r10']:.4f} ± {std_results['t2i_r10']:.4f}")
    print(f"    MRR:           {avg_results['t2i_mrr']:.4f} ± {std_results['t2i_mrr']:.4f}")
    print()
    print(f"  Image -> Text  (I2T)  [{eval_i2t} evaluated, {skip_i2t} skipped]")
    print(f"    Missed Top 10: {miss_i2t} (last run)")
    print(f"    Recall@1:      {avg_results['i2t_r1']:.4f} ± {std_results['i2t_r1']:.4f}")
    print(f"    Recall@5:      {avg_results['i2t_r5']:.4f} ± {std_results['i2t_r5']:.4f}")
    print(f"    Recall@10:     {avg_results['i2t_r10']:.4f} ± {std_results['i2t_r10']:.4f}")
    print(f"    MRR:           {avg_results['i2t_mrr']:.4f} ± {std_results['i2t_mrr']:.4f}")
    print("=" * 60)

    file_suffix = f"_{args.gallery_size}" if args.gallery_size else "_full"
    # Save to JSON
    json_path = os.path.join(os.path.dirname(__file__), "..", "..", f"results{file_suffix}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "avg": avg_results,
            "std": std_results
        }, f, indent=4)

    # Save to Markdown
    import datetime
    md_path = os.path.join(os.path.dirname(__file__), "..", "..", f"results{file_suffix}.md")
    
    # Extract versions
    import torch
    import transformers
    import faiss
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Evaluation Results\n\n")
        f.write(f"- **Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Model ID:** {settings.MODEL_ID}\n")
        f.write(f"- **Versions:** torch={torch.__version__}, transformers={transformers.__version__}, faiss={faiss.__version__}\n")
        f.write(f"- **Seeds:** {args.num_seeds} (starting from {args.seed})\n")
        f.write(f"- **Sample Size:** {eval_t2i}\n")
        f.write(f"- **Gallery Size:** {args.gallery_size if args.gallery_size else len(valid_img_ids)}\n\n")
        
        f.write(f"### Text -> Image (T2I)\n")
        f.write(f"- **Missed Top 10:** {miss_t2i} (last run)\n")
        f.write(f"- **Recall@1:** {avg_results['t2i_r1']:.4f} ± {std_results['t2i_r1']:.4f}\n")
        f.write(f"- **Recall@5:** {avg_results['t2i_r5']:.4f} ± {std_results['t2i_r5']:.4f}\n")
        f.write(f"- **Recall@10:** {avg_results['t2i_r10']:.4f} ± {std_results['t2i_r10']:.4f}\n")
        f.write(f"- **MRR:** {avg_results['t2i_mrr']:.4f} ± {std_results['t2i_mrr']:.4f}\n\n")

        f.write(f"### Image -> Text (I2T)\n")
        f.write(f"- **Missed Top 10:** {miss_i2t} (last run)\n")
        f.write(f"- **Recall@1:** {avg_results['i2t_r1']:.4f} ± {std_results['i2t_r1']:.4f}\n")
        f.write(f"- **Recall@5:** {avg_results['i2t_r5']:.4f} ± {std_results['i2t_r5']:.4f}\n")
        f.write(f"- **Recall@10:** {avg_results['i2t_r10']:.4f} ± {std_results['i2t_r10']:.4f}\n")
        f.write(f"- **MRR:** {avg_results['i2t_mrr']:.4f} ± {std_results['i2t_mrr']:.4f}\n")

    logger.info(f"Results saved to {json_path} and {md_path}")

if __name__ == "__main__":
    main()
