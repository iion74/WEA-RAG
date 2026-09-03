# Manuscript v4 artifact manifest

This manifest maps the numerical claims, tables, and result figures in
manuscript version 4 to the public artifacts and analysis code. Both HotpotQA
development settings contain 7,405 questions.

## Paper-to-artifact mapping

| Manuscript item | Frozen source | Public reproduction output |
|---|---|---|
| Figure 1: WEA-RAG pipeline | `rag_retrieve.py`, `rag_eval/wea_eval.py`, `docs/METHODS.md` | Conceptual method figure; no numerical output |
| Table 1 and Figure 2: Baseline, RAG, and WEA-RAG performance | `artifacts/metrics/*.json` | `artifacts/analysis/paper_results_reproduced.*`, `artifacts/figures/reproduced/main_performance.png` |
| Table 2: Hit@8, Recall@8, and Full coverage@8 | `artifacts/retrieval/*_top8.jsonl` | Retrieval section of `artifacts/analysis/paper_results_reproduced.*` |
| Figure 3: WEA activation, web-use, and query rates | `artifacts/predictions/*_wea_rag_*.jsonl` | `artifacts/figures/reproduced/wea_usage_*.png` |
| Figure 4: rescue, harm, net gain, and activated-case outcomes | Standalone RAG and WEA-RAG traces in `artifacts/predictions/` | `artifacts/figures/reproduced/wea_quality_*.png` |
| Figure 5: WEA-RAG latency | `avg_latency_sec` in `artifacts/metrics/*_wea_rag_*.json` | `artifacts/figures/reproduced/latency.png` |
| Table 3 and Figure 6: gate ablation | `artifacts/predictions/*_wea_rag_*.jsonl` | Gate section of `artifacts/analysis/ablation_reproduced.*`, `artifacts/figures/reproduced/gate_ablation.png` |
| Figure 7: selector ablation | Saved dual candidates and scores in the WEA-RAG traces | Selector section of `artifacts/analysis/ablation_reproduced.*`, `artifacts/figures/reproduced/selector_ablation.png` |
| Figure 8: retrieval-difficulty buckets | Coverage, missing terms, query status, and predictions in the traces | Difficulty section of `artifacts/analysis/ablation_reproduced.*`, `artifacts/figures/reproduced/retrieval_difficulty.png` |
| Paired bootstrap statistics | Aligned standalone RAG and final WEA-RAG traces | Bootstrap section of `artifacts/analysis/ablation_reproduced.*` |
| Final post-hoc selector settings | `artifacts/analysis/selector_tuning.json` | Applied by the stored final WEA-RAG traces and documented in `docs/METHODS.md` |
| Cached web evidence | `artifacts/web_cache/hotpot_shared_web_cache.jsonl` | Reused by `scripts/run_wea_rag.sh` in its default cache-only mode |

The original publication-ready result images are under
`artifacts/figures/published/`. The corresponding files under
`artifacts/figures/reproduced/` are generated from the released analysis code
and may differ in layout while representing the same artifact data.

## Completeness

- 18 aggregate metric files cover three systems, three models, and two splits.
- 12 prediction traces cover standalone RAG and WEA-RAG for every model/split
  pair; every trace contains 7,405 rows.
- Two retrieval traces contain 7,405 top-eight retrieval records each.
- The shared cache contains 4,867 query-result records and no API key field.
- Nine publication-ready result images and nine regenerated result images are
  included.
- Baseline per-example predictions were not retained in the original experiment;
  the release includes their six aggregate metric files and rerun scripts.

## Source-of-truth rules

The WEA-RAG per-example traces are canonical for WEA activation, query, web-use,
rescue, and harm counts. The aggregate Qwen distractor metrics file contains
an earlier cumulative web counter, so operational rates are recomputed from its
complete 7,405-row trace. Aggregate metric files remain the source for EM, F1,
and latency values.

## Excluded assets

Raw HotpotQA files, model weights, the generated 510,285-document corpus, the
FAISS index, local logs, API credentials, and manuscript editing files are not
committed. Download and reconstruction scripts are provided for assets that can
be rebuilt locally. Licensing details are in `THIRD_PARTY_NOTICES.md`.

## Verification

Run the release audit without model inference:

```bash
bash scripts/verify_release.sh
```

Regenerate the analysis tables and figures after preparing the dataset:

```bash
bash scripts/prepare_data.sh
bash scripts/reproduce_paper_analysis.sh
```
