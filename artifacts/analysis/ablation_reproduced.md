# Cached WEA-RAG Paper Experiments

## 1) Gate Ablation

| Model | Split | Policy | EM | F1 | Delta EM vs RAG | Delta F1 vs RAG | Rescue | Harm | Net | CallRate | GateTrueNoQuery |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | dev_distractor | RAG only | 53.44 | 69.18 | +0.00 | +0.00 | 0 | 0 | 0 | 0.00 | 0 |
| Qwen3-4B | dev_distractor | Gate t=0.78, m=2, time=on | 53.95 | 69.61 | +0.51 | +0.44 | 38 | 0 | 38 | 17.56 | 48 |
| Qwen3-4B | dev_distractor | Gate t=0.78, m=2, time=off | 53.94 | 69.60 | +0.50 | +0.43 | 37 | 0 | 37 | 17.50 | 0 |
| Qwen3-4B | dev_distractor | Gate t=0.78, m=4, time=on | 53.68 | 69.39 | +0.24 | +0.21 | 18 | 0 | 18 | 7.40 | 7 |
| Qwen3-4B | dev_distractor | Gate t=0.70, m=4, time=off | 53.54 | 69.27 | +0.11 | +0.09 | 8 | 0 | 8 | 4.63 | 0 |
| Qwen3-4B | dev_fullwiki | RAG only | 44.19 | 58.44 | +0.00 | +0.00 | 0 | 0 | 0 | 0.00 | 0 |
| Qwen3-4B | dev_fullwiki | Gate t=0.78, m=2, time=on | 44.98 | 59.35 | +0.80 | +0.91 | 133 | 74 | 59 | 25.56 | 48 |
| Qwen3-4B | dev_fullwiki | Gate t=0.78, m=2, time=off | 44.98 | 59.35 | +0.80 | +0.91 | 133 | 74 | 59 | 25.50 | 0 |
| Qwen3-4B | dev_fullwiki | Gate t=0.78, m=4, time=on | 44.33 | 58.63 | +0.15 | +0.19 | 48 | 37 | 11 | 10.93 | 6 |
| Qwen3-4B | dev_fullwiki | Gate t=0.70, m=4, time=off | 44.28 | 58.60 | +0.09 | +0.16 | 30 | 23 | 7 | 7.36 | 0 |
| Llama-3.2-3B | dev_distractor | RAG only | 47.39 | 60.70 | +0.00 | +0.00 | 0 | 0 | 0 | 0.00 | 0 |
| Llama-3.2-3B | dev_distractor | Gate t=0.78, m=2, time=on | 47.41 | 60.64 | +0.03 | -0.06 | 9 | 7 | 2 | 17.54 | 49 |
| Llama-3.2-3B | dev_distractor | Gate t=0.78, m=2, time=off | 47.41 | 60.64 | +0.03 | -0.06 | 9 | 7 | 2 | 17.50 | 0 |
| Llama-3.2-3B | dev_distractor | Gate t=0.78, m=4, time=on | 47.39 | 60.67 | +0.00 | -0.03 | 4 | 4 | 0 | 7.40 | 7 |
| Llama-3.2-3B | dev_distractor | Gate t=0.70, m=4, time=off | 47.39 | 60.68 | +0.00 | -0.02 | 3 | 3 | 0 | 4.63 | 0 |
| Llama-3.2-3B | dev_fullwiki | RAG only | 38.77 | 50.79 | +0.00 | +0.00 | 0 | 0 | 0 | 0.00 | 0 |
| Llama-3.2-3B | dev_fullwiki | Gate t=0.78, m=2, time=on | 39.05 | 51.09 | +0.28 | +0.30 | 111 | 90 | 21 | 25.58 | 47 |
| Llama-3.2-3B | dev_fullwiki | Gate t=0.78, m=2, time=off | 39.04 | 51.09 | +0.27 | +0.30 | 110 | 90 | 20 | 25.50 | 0 |
| Llama-3.2-3B | dev_fullwiki | Gate t=0.78, m=4, time=on | 38.85 | 50.90 | +0.08 | +0.10 | 40 | 34 | 6 | 10.91 | 7 |
| Llama-3.2-3B | dev_fullwiki | Gate t=0.70, m=4, time=off | 38.80 | 50.80 | +0.03 | +0.01 | 29 | 27 | 2 | 7.36 | 0 |
| MiniCPM3-4B | dev_distractor | RAG only | 47.36 | 61.15 | +0.00 | +0.00 | 0 | 0 | 0 | 0.00 | 0 |
| MiniCPM3-4B | dev_distractor | Gate t=0.78, m=2, time=on | 47.55 | 61.36 | +0.19 | +0.21 | 15 | 1 | 14 | 17.53 | 50 |
| MiniCPM3-4B | dev_distractor | Gate t=0.78, m=2, time=off | 47.55 | 61.36 | +0.19 | +0.21 | 15 | 1 | 14 | 17.50 | 0 |
| MiniCPM3-4B | dev_distractor | Gate t=0.78, m=4, time=on | 47.45 | 61.26 | +0.09 | +0.11 | 8 | 1 | 7 | 7.40 | 7 |
| MiniCPM3-4B | dev_distractor | Gate t=0.70, m=4, time=off | 47.44 | 61.22 | +0.08 | +0.07 | 6 | 0 | 6 | 4.63 | 0 |
| MiniCPM3-4B | dev_fullwiki | RAG only | 39.85 | 52.48 | +0.00 | +0.00 | 0 | 0 | 0 | 0.00 | 0 |
| MiniCPM3-4B | dev_fullwiki | Gate t=0.78, m=2, time=on | 40.39 | 52.98 | +0.54 | +0.50 | 107 | 67 | 40 | 25.55 | 49 |
| MiniCPM3-4B | dev_fullwiki | Gate t=0.78, m=2, time=off | 40.39 | 52.98 | +0.54 | +0.50 | 107 | 67 | 40 | 25.50 | 0 |
| MiniCPM3-4B | dev_fullwiki | Gate t=0.78, m=4, time=on | 39.97 | 52.54 | +0.12 | +0.07 | 44 | 35 | 9 | 10.91 | 7 |
| MiniCPM3-4B | dev_fullwiki | Gate t=0.70, m=4, time=off | 39.92 | 52.47 | +0.07 | -0.01 | 29 | 24 | 5 | 7.36 | 0 |

## 2) Selector Ablation

| Model | Split | Policy | EM | F1 | Delta EM vs tuned | Delta F1 vs tuned | Rescue | Harm | Net |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | dev_distractor | Stored tuned selector | 54.00 | 69.69 | +0.00 | +0.00 | 42 | 0 | 42 |
| Qwen3-4B | dev_distractor | Always keep RAG on dual cases | 53.50 | 69.27 | -0.50 | -0.42 | 5 | 0 | 5 |
| Qwen3-4B | dev_distractor | Always take WEA on dual cases | 53.99 | 69.77 | -0.01 | +0.08 | 70 | 29 | 41 |
| Qwen3-4B | dev_distractor | Simple selector m=0.22, WEA>=0.80, lock=0.88 | 54.00 | 69.69 | +0.00 | +0.00 | 42 | 0 | 42 |
| Qwen3-4B | dev_distractor | Simple selector m=0.00, WEA>=0.65, lock=0.88 | 53.99 | 69.69 | -0.01 | -0.00 | 42 | 1 | 41 |
| Qwen3-4B | dev_fullwiki | Stored tuned selector | 45.32 | 60.00 | +0.00 | +0.00 | 406 | 322 | 84 |
| Qwen3-4B | dev_fullwiki | Always keep RAG on dual cases | 44.65 | 59.14 | -0.68 | -0.86 | 369 | 335 | 34 |
| Qwen3-4B | dev_fullwiki | Always take WEA on dual cases | 45.48 | 60.31 | +0.16 | +0.31 | 428 | 332 | 96 |
| Qwen3-4B | dev_fullwiki | Simple selector m=0.22, WEA>=0.80, lock=0.88 | 45.32 | 59.99 | +0.00 | -0.01 | 406 | 322 | 84 |
| Qwen3-4B | dev_fullwiki | Simple selector m=0.00, WEA>=0.65, lock=0.88 | 45.29 | 59.99 | -0.03 | -0.00 | 406 | 324 | 82 |
| Llama-3.2-3B | dev_distractor | Stored tuned selector | 47.63 | 60.88 | +0.00 | +0.00 | 44 | 26 | 18 |
| Llama-3.2-3B | dev_distractor | Always keep RAG on dual cases | 47.59 | 60.86 | -0.04 | -0.02 | 40 | 25 | 15 |
| Llama-3.2-3B | dev_distractor | Always take WEA on dual cases | 46.75 | 60.13 | -0.88 | -0.74 | 71 | 118 | -47 |
| Llama-3.2-3B | dev_distractor | Simple selector m=0.22, WEA>=0.80, lock=0.88 | 47.63 | 60.88 | +0.00 | +0.00 | 44 | 26 | 18 |
| Llama-3.2-3B | dev_distractor | Simple selector m=0.00, WEA>=0.65, lock=0.88 | 47.62 | 60.86 | -0.01 | -0.01 | 44 | 27 | 17 |
| Llama-3.2-3B | dev_fullwiki | Stored tuned selector | 39.08 | 51.19 | +0.00 | +0.00 | 397 | 374 | 23 |
| Llama-3.2-3B | dev_fullwiki | Always keep RAG on dual cases | 38.95 | 51.05 | -0.14 | -0.14 | 387 | 374 | 13 |
| Llama-3.2-3B | dev_fullwiki | Always take WEA on dual cases | 38.30 | 50.43 | -0.78 | -0.76 | 414 | 449 | -35 |
| Llama-3.2-3B | dev_fullwiki | Simple selector m=0.22, WEA>=0.80, lock=0.88 | 39.08 | 51.19 | +0.00 | +0.00 | 397 | 374 | 23 |
| Llama-3.2-3B | dev_fullwiki | Simple selector m=0.00, WEA>=0.65, lock=0.88 | 39.04 | 51.16 | -0.04 | -0.02 | 396 | 376 | 20 |
| MiniCPM3-4B | dev_distractor | Stored tuned selector | 47.55 | 61.38 | +0.00 | +0.00 | 15 | 1 | 14 |
| MiniCPM3-4B | dev_distractor | Always keep RAG on dual cases | 47.36 | 61.16 | -0.19 | -0.22 | 0 | 0 | 0 |
| MiniCPM3-4B | dev_distractor | Always take WEA on dual cases | 47.75 | 61.60 | +0.20 | +0.22 | 44 | 15 | 29 |
| MiniCPM3-4B | dev_distractor | Simple selector m=0.22, WEA>=0.80, lock=0.88 | 47.54 | 61.37 | -0.01 | -0.01 | 14 | 1 | 13 |
| MiniCPM3-4B | dev_distractor | Simple selector m=0.00, WEA>=0.65, lock=0.88 | 47.54 | 61.38 | -0.01 | -0.00 | 15 | 2 | 13 |
| MiniCPM3-4B | dev_fullwiki | Stored tuned selector | 40.61 | 53.23 | +0.00 | +0.00 | 330 | 274 | 56 |
| MiniCPM3-4B | dev_fullwiki | Always keep RAG on dual cases | 40.27 | 52.81 | -0.34 | -0.42 | 307 | 276 | 31 |
| MiniCPM3-4B | dev_fullwiki | Always take WEA on dual cases | 41.12 | 53.95 | +0.51 | +0.72 | 376 | 282 | 94 |
| MiniCPM3-4B | dev_fullwiki | Simple selector m=0.22, WEA>=0.80, lock=0.88 | 40.61 | 53.23 | +0.00 | +0.00 | 330 | 274 | 56 |
| MiniCPM3-4B | dev_fullwiki | Simple selector m=0.00, WEA>=0.65, lock=0.88 | 40.58 | 53.23 | -0.03 | -0.01 | 331 | 277 | 54 |

## 3) Retrieval Difficulty Analysis

| Model | Split | Coverage | Missing | Count | RAG EM | WEA-RAG EM | Delta EM | RAG F1 | WEA-RAG F1 | Delta F1 | WEA rate |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | dev_distractor | high | 0-1 | 4041 | 55.90 | 55.98 | +0.07 | 71.39 | 71.48 | +0.08 | 0.00 |
| Qwen3-4B | dev_distractor | high | 2-3 | 948 | 52.32 | 52.53 | +0.21 | 69.55 | 69.78 | +0.23 | 0.11 |
| Qwen3-4B | dev_distractor | high | 4+ | 131 | 64.12 | 64.12 | +0.00 | 77.58 | 77.36 | -0.22 | 0.00 |
| Qwen3-4B | dev_distractor | low | 0-1 | 2 | 0.00 | 0.00 | +0.00 | 50.00 | 50.00 | +0.00 | 0.00 |
| Qwen3-4B | dev_distractor | low | 2-3 | 232 | 50.86 | 52.59 | +1.72 | 63.97 | 65.30 | +1.34 | 100.00 |
| Qwen3-4B | dev_distractor | low | 4+ | 343 | 42.86 | 45.19 | +2.33 | 58.46 | 60.44 | +1.99 | 100.00 |
| Qwen3-4B | dev_distractor | mid | 0-1 | 257 | 49.03 | 49.03 | +0.00 | 63.58 | 63.86 | +0.28 | 0.00 |
| Qwen3-4B | dev_distractor | mid | 2-3 | 1144 | 49.74 | 51.05 | +1.31 | 66.47 | 67.59 | +1.13 | 47.03 |
| Qwen3-4B | dev_distractor | mid | 4+ | 307 | 51.47 | 54.72 | +3.26 | 66.13 | 69.06 | +2.93 | 60.59 |
| Qwen3-4B | dev_fullwiki | high | 0-1 | 3351 | 46.67 | 47.15 | +0.48 | 61.15 | 62.15 | +1.01 | 0.00 |
| Qwen3-4B | dev_fullwiki | high | 2-3 | 890 | 45.62 | 46.40 | +0.79 | 62.28 | 62.52 | +0.24 | 0.00 |
| Qwen3-4B | dev_fullwiki | high | 4+ | 115 | 55.65 | 56.52 | +0.87 | 70.01 | 71.73 | +1.72 | 0.00 |
| Qwen3-4B | dev_fullwiki | low | 0-1 | 6 | 50.00 | 50.00 | +0.00 | 66.67 | 68.25 | +1.59 | 0.00 |
| Qwen3-4B | dev_fullwiki | low | 2-3 | 363 | 37.19 | 39.39 | +2.20 | 51.09 | 54.00 | +2.91 | 100.00 |
| Qwen3-4B | dev_fullwiki | low | 4+ | 545 | 36.15 | 37.43 | +1.28 | 47.60 | 49.76 | +2.16 | 100.00 |
| Qwen3-4B | dev_fullwiki | mid | 0-1 | 343 | 38.48 | 37.03 | -1.46 | 50.97 | 50.26 | -0.72 | 0.00 |
| Qwen3-4B | dev_fullwiki | mid | 2-3 | 1435 | 41.53 | 45.02 | +3.48 | 55.30 | 59.27 | +3.97 | 51.99 |
| Qwen3-4B | dev_fullwiki | mid | 4+ | 357 | 49.02 | 49.02 | +0.00 | 63.43 | 63.62 | +0.18 | 66.95 |
| Llama-3.2-3B | dev_distractor | high | 0-1 | 4041 | 47.69 | 48.06 | +0.37 | 61.31 | 61.74 | +0.43 | 0.00 |
| Llama-3.2-3B | dev_distractor | high | 2-3 | 948 | 48.63 | 48.52 | -0.11 | 64.54 | 64.42 | -0.12 | 0.11 |
| Llama-3.2-3B | dev_distractor | high | 4+ | 131 | 54.20 | 54.20 | +0.00 | 63.37 | 63.37 | +0.00 | 0.00 |
| Llama-3.2-3B | dev_distractor | low | 0-1 | 2 | 50.00 | 50.00 | +0.00 | 50.00 | 50.00 | +0.00 | 0.00 |
| Llama-3.2-3B | dev_distractor | low | 2-3 | 232 | 43.97 | 44.40 | +0.43 | 53.54 | 53.71 | +0.16 | 100.00 |
| Llama-3.2-3B | dev_distractor | low | 4+ | 343 | 44.02 | 44.02 | +0.00 | 56.01 | 55.49 | -0.52 | 100.00 |
| Llama-3.2-3B | dev_distractor | mid | 0-1 | 257 | 47.08 | 47.47 | +0.39 | 54.06 | 54.26 | +0.19 | 0.00 |
| Llama-3.2-3B | dev_distractor | mid | 2-3 | 1144 | 45.80 | 45.98 | +0.17 | 58.63 | 58.50 | -0.13 | 46.94 |
| Llama-3.2-3B | dev_distractor | mid | 4+ | 307 | 49.19 | 49.19 | +0.00 | 63.67 | 63.45 | -0.22 | 60.59 |
| Llama-3.2-3B | dev_fullwiki | high | 0-1 | 3351 | 39.54 | 39.54 | +0.00 | 52.34 | 52.64 | +0.30 | 0.00 |
| Llama-3.2-3B | dev_fullwiki | high | 2-3 | 890 | 41.35 | 40.79 | -0.56 | 55.64 | 54.70 | -0.94 | 0.11 |
| Llama-3.2-3B | dev_fullwiki | high | 4+ | 115 | 43.48 | 46.96 | +3.48 | 55.39 | 58.55 | +3.16 | 0.00 |
| Llama-3.2-3B | dev_fullwiki | low | 0-1 | 6 | 33.33 | 66.67 | +33.33 | 55.56 | 83.33 | +27.78 | 0.00 |
| Llama-3.2-3B | dev_fullwiki | low | 2-3 | 363 | 32.78 | 33.61 | +0.83 | 43.34 | 44.10 | +0.76 | 100.00 |
| Llama-3.2-3B | dev_fullwiki | low | 4+ | 545 | 35.96 | 36.33 | +0.37 | 45.20 | 45.33 | +0.13 | 100.00 |
| Llama-3.2-3B | dev_fullwiki | mid | 0-1 | 343 | 41.40 | 40.23 | -1.17 | 47.97 | 47.41 | -0.56 | 0.00 |
| Llama-3.2-3B | dev_fullwiki | mid | 2-3 | 1435 | 37.21 | 38.12 | +0.91 | 48.33 | 48.98 | +0.65 | 52.06 |
| Llama-3.2-3B | dev_fullwiki | mid | 4+ | 357 | 37.82 | 40.06 | +2.24 | 51.34 | 54.46 | +3.12 | 66.67 |
| MiniCPM3-4B | dev_distractor | high | 0-1 | 4041 | 47.59 | 47.59 | +0.00 | 61.60 | 61.61 | +0.01 | 0.00 |
| MiniCPM3-4B | dev_distractor | high | 2-3 | 948 | 47.26 | 47.26 | +0.00 | 62.58 | 62.60 | +0.01 | 0.00 |
| MiniCPM3-4B | dev_distractor | high | 4+ | 131 | 52.67 | 52.67 | +0.00 | 69.23 | 69.26 | +0.04 | 0.00 |
| MiniCPM3-4B | dev_distractor | low | 0-1 | 2 | 0.00 | 0.00 | +0.00 | 58.33 | 58.33 | +0.00 | 0.00 |
| MiniCPM3-4B | dev_distractor | low | 2-3 | 232 | 43.10 | 44.40 | +1.29 | 53.98 | 55.21 | +1.23 | 100.00 |
| MiniCPM3-4B | dev_distractor | low | 4+ | 343 | 44.90 | 46.65 | +1.75 | 55.67 | 57.19 | +1.52 | 100.00 |
| MiniCPM3-4B | dev_distractor | mid | 0-1 | 257 | 51.75 | 51.75 | +0.00 | 61.58 | 61.63 | +0.04 | 0.00 |
| MiniCPM3-4B | dev_distractor | mid | 2-3 | 1144 | 46.33 | 46.77 | +0.44 | 60.45 | 61.00 | +0.55 | 46.94 |
| MiniCPM3-4B | dev_distractor | mid | 4+ | 307 | 48.86 | 48.86 | +0.00 | 61.12 | 61.80 | +0.67 | 60.59 |
| MiniCPM3-4B | dev_fullwiki | high | 0-1 | 3351 | 40.91 | 41.30 | +0.39 | 53.97 | 54.71 | +0.74 | 0.00 |
| MiniCPM3-4B | dev_fullwiki | high | 2-3 | 890 | 41.24 | 41.35 | +0.11 | 56.55 | 56.29 | -0.26 | 0.00 |
| MiniCPM3-4B | dev_fullwiki | high | 4+ | 115 | 53.91 | 49.57 | -4.35 | 63.81 | 62.46 | -1.34 | 0.00 |
| MiniCPM3-4B | dev_fullwiki | low | 0-1 | 6 | 33.33 | 50.00 | +16.67 | 52.22 | 62.70 | +10.48 | 0.00 |
| MiniCPM3-4B | dev_fullwiki | low | 2-3 | 363 | 35.81 | 36.91 | +1.10 | 46.00 | 47.87 | +1.88 | 100.00 |
| MiniCPM3-4B | dev_fullwiki | low | 4+ | 545 | 37.06 | 37.98 | +0.92 | 47.30 | 47.16 | -0.14 | 100.00 |
| MiniCPM3-4B | dev_fullwiki | mid | 0-1 | 343 | 39.07 | 37.61 | -1.46 | 50.20 | 47.36 | -2.84 | 0.00 |
| MiniCPM3-4B | dev_fullwiki | mid | 2-3 | 1435 | 36.72 | 39.37 | +2.65 | 48.69 | 50.89 | +2.20 | 51.99 |
| MiniCPM3-4B | dev_fullwiki | mid | 4+ | 357 | 43.70 | 44.82 | +1.12 | 56.58 | 58.37 | +1.80 | 66.67 |

## 4) Bootstrap Significance

| Model | Split | Delta EM | EM 95% CI | p(<=0) | Delta F1 | F1 95% CI | p(<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | dev_distractor | 0.57 | [0.41, 0.74] | 0.0002 | 0.51 | [0.36, 0.67] | 0.0002 |
| Qwen3-4B | dev_fullwiki | 1.13 | [0.45, 1.85] | 0.0008 | 1.56 | [0.91, 2.21] | 0.0002 |
| Llama-3.2-3B | dev_distractor | 0.24 | [0.03, 0.46] | 0.0178 | 0.18 | [-0.02, 0.38] | 0.0400 |
| Llama-3.2-3B | dev_fullwiki | 0.31 | [-0.43, 1.04] | 0.2092 | 0.39 | [-0.34, 1.12] | 0.1456 |
| MiniCPM3-4B | dev_distractor | 0.19 | [0.08, 0.30] | 0.0004 | 0.23 | [0.12, 0.35] | 0.0004 |
| MiniCPM3-4B | dev_fullwiki | 0.76 | [0.12, 1.42] | 0.0112 | 0.76 | [0.13, 1.40] | 0.0094 |

## Notes

- Gate ablation is an offline counterfactual over the cached traced run and is valid only for policies nested within the observed trigger set.
- GateTrueNoQuery means the gate would fire under that policy, but the stored run did not issue a web query for that sample; those cases fall back to the RAG answer.
- Selector ablation reuses stored dual predictions and scores from cached final prediction files.
- Bootstrap significance compares RAG with stored tuned WEA-RAG over the same question set.
