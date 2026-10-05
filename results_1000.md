# Evaluation Results

- **Date:** 2026-10-05 15:22:21
- **Model ID:** openai/clip-vit-base-patch32
- **Versions:** torch=2.11.0+cpu, transformers=5.17.0, faiss=1.15.1
- **Seeds:** 3 (starting from 42)
- **Sample Size:** 1000
- **Gallery Size:** 1000

### Text -> Image (T2I)
- **Missed Top 10:** 90 (last run)
- **Recall@1:** 0.5813 ± 0.0162
- **Recall@5:** 0.8337 ± 0.0012
- **Recall@10:** 0.9063 ± 0.0090
- **MRR:** 0.6901 ± 0.0107

### Image -> Text (I2T)
- **Missed Top 10:** 20 (last run)
- **Recall@1:** 0.8040 ± 0.0108
- **Recall@5:** 0.9567 ± 0.0054
- **Recall@10:** 0.9807 ± 0.0041
- **MRR:** 0.8688 ± 0.0087
