# Method and experiment settings

This document maps the released implementation to the method and settings
reported in manuscript version 4. Code, command-line options, and public
artifact fields use WEA terminology.

## Evaluation data and systems

Experiments use all 7,405 questions in each HotpotQA development setting:
`distractor` and `fullwiki`. Three instruction-tuned models are evaluated:

- `Qwen/Qwen3-4B-Instruct-2507`
- `meta-llama/Llama-3.2-3B-Instruct`
- `openbmb/MiniCPM3-4B`

The compared systems are:

- **Baseline:** a provided-context reader. It receives the context distributed
  with each dataset record and is not a closed-book baseline.
- **RAG:** a reader supplied with the top-eight documents produced by the
  split-aware retrieval pipeline.
- **WEA-RAG:** the RAG path followed by retrieval-sufficiency assessment,
  selective web evidence augmentation, and guarded answer selection.

## Retrieval corpus and split-aware retrieval

The experiment corpus contains 510,285 documents collected from the context
documents in HotpotQA train, dev-distractor, and dev-fullwiki. Documents are
deduplicated by title, and each stored text is limited to 2,000 characters.
This is the corpus used by the experiment and is not a complete Wikipedia dump.

For `distractor`, candidates are restricted to the context documents provided
with each question. These candidates are reranked without global dense search.
For `fullwiki`, the system searches the constructed corpus with
`BAAI/bge-large-en-v1.5`, retrieves up to 200 dense candidates, and reranks up
to 120 candidates with `BAAI/bge-reranker-large`. Heuristic question
decomposition produces at most three queries. Two-hop expansion uses the first
two retrieved documents and generates at most two bridge queries per document.

For candidate document `i`, the final retrieval score is:

```text
s_doc(i) = s_rerank(i)
           + 0.05 * norm(s_dense(i))
           + 0.20 * norm(g(i))
           + 0.10 * norm(q(i))
```

`s_rerank` is the raw cross-encoder score. The other three terms are min-max
normalized within the current candidate set. `g(i)` is the maximum
entity-overlap link between candidate `i` and four seed documents. Seeds are
the four candidates with the greatest question-entity overlap, or the first
four candidates if every overlap is zero. `q(i)` is question-document entity
overlap. In `distractor`, no dense search is performed, so the normalized dense
term is zero.

The generation context concatenates the titles and texts of the top-eight
documents and truncates the result to 16,000 characters. The retrieval traces
also store up to 16 evidence sentences with at most two sentences per document
for analysis. Those compressed sentence lists are not used as the generation
context in the reported runs.

## Retrieval sufficiency gate

Question and context strings are lowercased and tokenized with the ASCII
alphanumeric pattern `[A-Za-z0-9]+`; one-character tokens are discarded.
Question stopwords are removed. Coverage is the exact surface-token overlap:

```text
Cov(Q, C_RAG) = |T_f(Q) intersect T(C_RAG)| / |T_f(Q)|
```

No synonym expansion or semantic similarity is used. Missing terms initially
contain the filtered question tokens absent from the RAG context. If every
token of a capitalized question phrase is not present, tokens from that phrase
are also added to the missing-term set.

The initial WEA gate passes a question when any of the following holds:

- the RAG context is empty;
- coverage is below `0.78` and at least two missing terms remain;
- the question is time-sensitive and either of the preceding two risk signals
  is present.

Time sensitivity is detected by a fixed regular expression covering terms such
as `current`, `latest`, `recent`, `as of`, `this year`, `ongoing`, and `now`.

Questions passing the initial gate undergo a second check before web search.
WEA proceeds for an empty or refusal RAG answer, or under the implemented
numeric-format, low answer-score, weak token-support, and missing-clue rules.
These checks use coverage `0.80`, answer score `0.45`, token support `0.20`, and
the same minimum of two missing terms where applicable. The exact branches are
implemented in `decide_wea_refine_needed` in `rag_eval/wea_eval.py`.

## Web evidence augmentation

Web retrieval uses the SerpAPI Google Search interface and keeps up to eight
organic results represented by title and snippet. The query strategy first
tries the original question and then the question followed by up to five sorted
missing terms. Web evidence is limited to 6,000 characters, and the combined
RAG and web context is limited to 20,000 characters.

The WEA answer candidate is generated from the question, the initial RAG
answer, the RAG context, the retrieved web evidence, and the missing terms. In
the reported final evaluation, all requested search results were served from
the frozen shared cache; no live SerpAPI request was made during final scoring.

## Answer scoring and guarded selection

For an answer `A` and its evidence context `C`, the rule-based alignment score
uses token support, direct phrase occurrence, refusal detection, numeric-answer
format, invalid yes/no output, and answer length:

```text
S(A, C) = 1.6 * token_support
          + 0.6 * phrase_match
          - 1.0 * refusal
          +/- 0.25 * numeric_format
          - 0.1 * invalid_yes_no
          - min(0.7, 0.06 * (word_count - 8))  if word_count > 8
```

The online selector used while generating the dual traces required a WEA score
of at least `0.65`, token support of at least `0.55`, direct context support, a
RAG protection score of `0.88`, a score margin, and conflict guards.

The manuscript's final WEA-RAG answers were reselected from the saved RAG and
WEA candidates by a post-hoc selector. It retained the minimum WEA score
`0.65`, the RAG protection score `0.88`, and used these model/split margins:

```text
delta = S_WEA - S_RAG
select WEA when S_WEA >= 0.65 and delta >= margin,
except keep RAG when S_RAG >= 0.88 and delta < margin + 0.05
```

## Generation and runtime environment

Standalone Baseline and RAG runs used a maximum of 32 newly generated tokens.
The WEA-RAG run used 64 for its internal RAG draft and WEA candidate. The random
seed was `13`.

The reported environment was Ubuntu 24.04.4 LTS, Python 3.12.3, PyTorch
2.6.0+cu124, CUDA 12.4, Transformers 4.57.1, Sentence Transformers 5.2.0,
FAISS 1.11.0, and Accelerate 1.12.0 on an NVIDIA GeForce RTX 3090 Ti with 24 GB
of VRAM.

The model snapshots present in the original experiment environment resolved to
the following Hugging Face revisions:

| Model | Revision |
|---|---|
| Qwen3-4B-Instruct-2507 | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Llama-3.2-3B-Instruct | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| MiniCPM3-4B | `d6b14ddaefdb11c624dd75c3c779549bc90b08cb` |
| bge-large-en-v1.5 | `d4aa6901d3a41ba39fb536a557fa166f842b0e09` |
| bge-reranker-large | `55611d7bca2a7133960a6d3b71e083071bbfc312` |

## Evaluation

Answer quality is measured with HotpotQA-style Exact Match and token-level F1.
Retrieval is measured with Hit@8, Recall@8, and Full coverage@8. Auxiliary
analyses report web-query and web-use rates, cache-hit rate, latency, rescue,
harm, net gain, gate and selector ablations, and retrieval-difficulty buckets.
The ablations operate on saved cached traces and are not fresh live-web runs.
