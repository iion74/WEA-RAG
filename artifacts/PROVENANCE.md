# Artifact provenance

Public filenames are concise aliases for the original experiment outputs. The
files are grouped by provided-context Baseline, standalone RAG, final WEA-RAG,
selector tuning, and offline ablation analysis.

The final WEA-RAG traces contain both the original RAG candidate and the
web-evidence-augmented candidate where WEA was performed. Their public fields
are `dual_rag_pred`, `dual_wea_pred`, `dual_rag_score`, and `dual_wea_score`.
`dual_wea_pred` corresponds to the WEA candidate in the manuscript.

The online run that generated the dual candidates used a WEA minimum score of
0.65, minimum token support of 0.55, direct context-match enforcement, a RAG
lock score of 0.88, dynamic margins, and conflict guards. The final reported
answers were then reselected from those stored candidates by the post-hoc tuned
selector. This post-hoc selector used only a WEA minimum score of 0.65, a RAG
lock score of 0.88, and the following fixed model/split margins:

| Model | Distractor | Fullwiki |
|---|---:|---:|
| Qwen3-4B-Instruct-2507 | 0.20 | 0.08 |
| Llama-3.2-3B-Instruct | 0.20 | 0.20 |
| MiniCPM3-4B | 0.03 | 0.20 |

The per-example prediction traces are canonical for operational counts. The
aggregate Qwen distractor metrics file contains lower cumulative web counters
than its complete 7,405-row final trace (`1,281` queries versus `1,300`, and
`1,270` web-used rows versus `1,289`). Reproduced usage statistics are therefore
computed from the final trace. The aggregate metric values are retained for
auditability.

The standalone RAG traces and WEA-RAG traces were generated in separate runs.
Standalone RAG used `max_new_tokens=32`; the WEA-RAG run used
`max_new_tokens=64` for both its internal RAG draft and WEA candidate. Final
predictions can therefore differ even on rows where no web query was issued.
The manuscript comparison is a pipeline-level comparison, not an isolated
causal estimate of WEA.

Post-hoc selector parameters were chosen independently for each model/split by
maximizing `EM + F1` on that split's complete HotpotQA dev labels. No disjoint
held-out split was used for this tuning. The tuned scores are therefore
descriptive dev-set results and may be optimistic as estimates of generalization.
