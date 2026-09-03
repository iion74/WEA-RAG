# Reproduced paper results

| Model | Split | Baseline EM/F1 | RAG EM/F1 | WEA-RAG EM/F1 | Delta vs. RAG EM/F1 |
|---|---|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 | distractor | 47.86/64.35 | 53.44/69.18 | 54.00/69.69 | +0.57/+0.51 |
| Qwen3-4B-Instruct-2507 | fullwiki | 27.60/38.91 | 44.19/58.44 | 45.32/60.00 | +1.13/+1.56 |
| Llama-3.2-3B-Instruct | distractor | 45.12/58.21 | 47.39/60.70 | 47.63/60.88 | +0.24/+0.18 |
| Llama-3.2-3B-Instruct | fullwiki | 28.66/37.66 | 38.77/50.79 | 39.08/51.19 | +0.31/+0.39 |
| MiniCPM3-4B | distractor | 40.49/54.85 | 47.36/61.15 | 47.55/61.38 | +0.19/+0.23 |
| MiniCPM3-4B | fullwiki | 28.09/39.23 | 39.85/52.48 | 40.61/53.23 | +0.76/+0.76 |

## Retrieval quality

| Split | Hit@8 | Recall@8 | Full coverage@8 |
|---|---:|---:|---:|
| distractor | 99.96 | 98.69 | 97.42 |
| fullwiki | 98.72 | 87.06 | 75.41 |
