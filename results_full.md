# Evaluation Results

- **Date:** 2026-10-05 15:35:36
- **Model ID:** openai/clip-vit-base-patch32
- **Versions:** torch=2.11.0+cpu, transformers=5.17.0, faiss=1.15.1
- **Seeds:** 3 (starting from 42)
- **Sample Size:** 1000
- **Gallery Size:** 31783

### Text -> Image (T2I)
- **Missed Top 10:** 482 (last run)
- **Recall@1:** 0.2233 ± 0.0161
- **Recall@5:** 0.4203 ± 0.0207
- **Recall@10:** 0.5177 ± 0.0167
- **MRR:** 0.3078 ± 0.0165

### Image -> Text (I2T)
- **Missed Top 10:** 262 (last run)
- **Recall@1:** 0.4083 ± 0.0159
- **Recall@5:** 0.6423 ± 0.0165
- **Recall@10:** 0.7347 ± 0.0133
- **MRR:** 0.5096 ± 0.0157
