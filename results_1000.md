# Evaluation Results

- **Date:** 2026-10-05 18:02:29
- **Model ID:** openai/clip-vit-base-patch32
- **Versions:** torch=2.11.0+cpu, transformers=5.17.0, faiss=1.15.1
- **Seeds:** 3 (starting from 42)
- **Sample Size:** 1000
- **Gallery Size:** 1000

### Text -> Image (T2I)
- **Missed Top 10:** 106 (last run)
- **Recall@1:** 0.5677 ± 0.0068
- **Recall@5:** 0.8253 ± 0.0083
- **Recall@10:** 0.8920 ± 0.0043
- **MRR:** 0.6777 ± 0.0061

### Image -> Text (I2T)
- **Missed Top 10:** 24 (last run)
- **Recall@1:** 0.7910 ± 0.0086
- **Recall@5:** 0.9420 ± 0.0043
- **Recall@10:** 0.9790 ± 0.0022
- **MRR:** 0.8588 ± 0.0070
