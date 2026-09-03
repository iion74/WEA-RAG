# Third-party notices

## HotpotQA

The frozen retrieval and prediction artifacts contain identifiers, questions,
answers, titles, or evidence text derived from HotpotQA. HotpotQA is distributed
under the Creative Commons Attribution-ShareAlike 4.0 International license.

- Project: <https://hotpotqa.github.io/>
- License: <https://creativecommons.org/licenses/by-sa/4.0/>
- Citation: Zhilin Yang et al., "HotpotQA: A Dataset for Diverse, Explainable
  Multi-hop Question Answering," EMNLP 2018.

## Model weights

No model weights are included. Qwen3-4B-Instruct-2507,
Llama-3.2-3B-Instruct, MiniCPM3-4B, BGE-large-en-v1.5, and
BGE-reranker-large must be obtained from their publishers and are governed by
their respective licenses and access conditions.

## Web search cache

`artifacts/web_cache/hotpot_shared_web_cache.jsonl` contains search queries,
URLs, titles, and short snippets returned through SerpAPI. These records are
included only to document and reproduce the cache-available evaluation. The
repository's Apache-2.0 license does not grant rights to third-party page
content or search-result snippets. Repository maintainers and users are
responsible for reviewing SerpAPI's current terms and the terms of the
originating sources before redistribution or reuse.
