# Evaluation Results

- **Date:** 2026-10-05 15:13:06
- **Model ID:** openai/clip-vit-base-patch32
- **Versions:** torch=2.11.0+cpu, transformers=5.17.0, faiss=1.15.1
- **Seeds:** 3 (starting from 42)
- **Sample Size:** 1000
- **Gallery Size:** 31783

### Text -> Image (T2I)
- **Missed Top 10:** 519 (last run)
- **Recall@1:** 0.2277 ± 0.0132
- **Recall@5:** 0.4153 ± 0.0045
- **Recall@10:** 0.5053 ± 0.0192
- **MRR:** 0.3101 ± 0.0047

### Image -> Text (I2T)
- **Missed Top 10:** 263 (last run)
- **Recall@1:** 0.4047 ± 0.0038
- **Recall@5:** 0.6507 ± 0.0059
- **Recall@10:** 0.7393 ± 0.0071
- **MRR:** 0.5106 ± 0.0033
