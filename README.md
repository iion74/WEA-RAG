# A Selective Web Evidence Augmentation Framework Based on Retrieval Sufficiency Assessment for Multi-Hop Question Answering
<img width="1257" height="650" alt="Image" src="https://github.com/user-attachments/assets/7ab23947-274e-4918-90f2-53a94342c482" />
Official reproducibility package for **A Selective Web Evidence Augmentation
Framework Based on Retrieval Sufficiency Assessment for Multi-Hop Question
Answering**.


WEA-RAG first generates an answer from split-aware retrieval. A rule-based gate
then uses question-context coverage and missing terms to decide whether web
evidence is needed. When the WEA path is activated, evidence returned by Google
Search through SerpAPI is combined with the RAG context to generate an
alternative answer. A guarded selector chooses between the original RAG answer
and the WEA candidate. Public commands, configuration names, and artifact
fields use WEA terminology consistently.

Detailed documentation:

- [Method and experiment settings](docs/METHODS.md)
- [Reported results and artifact mapping](docs/RESULTS.md)
- [Manuscript v4 artifact manifest](docs/ARTIFACT_MANIFEST.md)
- [Artifact provenance](artifacts/PROVENANCE.md)

## Reported full-dev results

| Model | Split | Baseline EM/F1 | RAG EM/F1 | WEA-RAG EM/F1 | Delta vs. RAG |
|---|---|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 | Distractor | 47.86/64.35 | 53.44/69.18 | 54.00/69.69 | +0.57/+0.51 |
| Qwen3-4B-Instruct-2507 | Fullwiki | 27.60/38.91 | 44.19/58.44 | 45.32/60.00 | +1.13/+1.56 |
| Llama-3.2-3B-Instruct | Distractor | 45.12/58.21 | 47.39/60.70 | 47.63/60.88 | +0.24/+0.18 |
| Llama-3.2-3B-Instruct | Fullwiki | 28.66/37.66 | 38.77/50.79 | 39.08/51.19 | +0.31/+0.39 |
| MiniCPM3-4B | Distractor | 40.49/54.85 | 47.36/61.15 | 47.55/61.38 | +0.19/+0.23 |
| MiniCPM3-4B | Fullwiki | 28.09/39.23 | 39.85/52.48 | 40.61/53.23 | +0.76/+0.76 |

The baseline is a provided-context reader, not a closed-book model. Final
WEA-RAG values were obtained by applying a post-hoc tuned selector to stored
dual prediction traces. All reported web queries were served from a shared
cache; no live SerpAPI request occurred during the final tuned evaluation.

The exact generation context, generation limits, available trace granularity,
and selector fitting protocol are documented in [METHODS](docs/METHODS.md) and
[artifact provenance](artifacts/PROVENANCE.md).

## Repository layout

```text
.
├── analysis/              # result summaries, ablations, validation, figures
├── artifacts/             # frozen paper metrics, traces, retrieval, web cache
├── basic_eval/            # provided-context baseline evaluators
├── configs/               # prompt and exact paper settings
├── data/                  # downloaded HotpotQA files, ignored by Git
├── docs/                  # method and result documentation
├── rag_eval/              # RAG and selective WEA evaluators
├── scripts/               # preparation and reproduction entry points
├── rag_build_corpus.py
├── rag_build_index.py
└── rag_retrieve.py
```

The repository excludes model weights, raw HotpotQA files, the generated
510,285-document corpus, the approximately 2 GB FAISS index, temporary logs,
unreported benchmark trials, and manuscript editing files.

The 510,285-document retrieval corpus used in the experiment is not a complete
Wikipedia dump. It was constructed from the context documents in HotpotQA
train, dev-distractor, and dev-fullwiki, deduplicated by title, with each text
limited to 2,000 characters. Distractor retrieval was restricted to each
question's provided candidates; fullwiki retrieval searched this constructed
corpus without that per-question restriction.

## Environment

The reported experiments used Ubuntu 24.04.4 LTS, Python 3.12.3, PyTorch
2.6.0+cu124, CUDA 12.4, Transformers 4.57.1, Sentence Transformers 5.2.0,
FAISS 1.11.0, and Accelerate 1.12.0. The experiment machine used an NVIDIA
GeForce RTX 3090 Ti with 24 GB of VRAM.

Create an environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The PyTorch wheel in `requirements.txt` does not force a CUDA wheel source.
Install the PyTorch build appropriate for the local CUDA driver if necessary.
Llama access may require accepting the model license on Hugging Face and
logging in before the first run.

## Reproduce paper analysis

This path recomputes the paper tables, bootstrap statistics, gate and selector
ablations, retrieval-difficulty buckets, and figures from the frozen prediction
traces. Model inference is not required.

```bash
bash scripts/verify_release.sh
```

This verifies the frozen artifacts and runs the unit tests without downloading
the dataset or loading a language model. To regenerate all tables, analyses,
and figures, run:

```bash
bash scripts/prepare_data.sh
bash scripts/reproduce_paper_analysis.sh
```

Outputs are written to:

- `artifacts/analysis/paper_results_reproduced.md`
- `artifacts/analysis/ablation_reproduced.md`
- `artifacts/figures/reproduced/`

Run the unit tests separately with:

```bash
python -m unittest discover -s tests -v
```

The ablations evaluate alternative policies over the stored WEA-RAG traces and
do not issue new web requests.

## Re-run model inference

Download and convert HotpotQA, then reconstruct the corpus in the same document
insertion order used by the reported experiments:

```bash
bash scripts/prepare_data.sh
bash scripts/build_corpus.sh
```

Run the provided-context baseline and RAG using the frozen top-8 retrieval
traces from the paper:

```bash
bash scripts/run_baseline_rag.sh
```

Run WEA-RAG using the frozen shared web cache:

```bash
bash scripts/run_wea_rag.sh
```

To repeat the manuscript's post-hoc selector search on newly generated dual
traces, run:

```bash
python analysis/tune_selector.py \
  --input_dir predictions \
  --metrics_dir metrics \
  --output_dir predictions/posthoc_tuned \
  --report_json metrics/selector_tuning.json \
  --report_md metrics/selector_tuning.md
```

This command repeats the full-development selector fitting protocol reported in
manuscript version 4.

Limit a smoke test to one model and a small number of questions:

```bash
MODEL_KEYS=qwen SPLITS=distractor LIMIT=20 bash scripts/run_baseline_rag.sh
MODEL_KEYS=qwen SPLITS=distractor LIMIT=20 bash scripts/run_wea_rag.sh
```

Fresh inference can differ slightly across GPU architectures and dependency
builds even with a fixed seed. The files under `artifacts/` are the source of
truth for the values reported in the manuscript.

## Rebuild retrieval

The frozen retrieval traces are included because they determine the exact
reported RAG inputs. To rebuild retrieval with the released implementation:

```bash
bash scripts/build_index.sh
bash scripts/rebuild_retrieval.sh
```

The dense index is generated locally and intentionally excluded from Git.
Rebuilt ranks may vary with model and numerical-library versions; use
`artifacts/retrieval/` when reproducing the published tables.

The graph-connectivity calculation uses
`max(4, 2 * HOP_TITLES_K)` seed documents. With the paper setting
`HOP_TITLES_K=2`, this is four seed documents.

## Live web search

The default WEA command is cache-only. For a fresh web run, provide a SerpAPI
key through the environment and explicitly enable live search:

```bash
export SERPAPI_API_KEY="your-key"
LIVE_WEB=1 bash scripts/run_wea_rag.sh
```

Never commit an API key. Live results are written to
`rag_cache/live_web_cache.jsonl`, leaving the frozen paper cache unchanged.
They are time-varying and are therefore not expected to match the
cache-available paper results exactly.

## Data and licenses

The code in this repository is released under Apache-2.0. HotpotQA and fields in
the frozen artifacts derived from HotpotQA are covered separately by CC BY-SA
4.0. Model weights are not redistributed and remain subject to their respective
licenses. The shared search cache contains query strings, source URLs, titles,
and search snippets collected for research reproducibility. Those third-party
materials are not covered by this repository's Apache-2.0 license. See
`THIRD_PARTY_NOTICES.md` before redistributing the artifacts.
