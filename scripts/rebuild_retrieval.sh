#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
source "${ROOT_DIR}/configs/paper.env"

common_args=(
  --corpus_file "${ROOT_DIR}/rag_corpus/docs.jsonl"
  --index_dir "${ROOT_DIR}/rag_index"
  --top_k "${TOP_K}"
  --dense_k "${DENSE_K}"
  --rerank_k "${RERANK_K}"
  --batch_size "${RETRIEVAL_BATCH_SIZE}"
  --device "${DEVICE}"
  --two_hop
  --hop_titles_k "${HOP_TITLES_K}"
  --subquestion_mode heuristic
  --subquestion_max "${SUBQUESTION_MAX}"
  --bridge_query_per_doc "${BRIDGE_QUERY_PER_DOC}"
  --dense_weight "${DENSE_WEIGHT}"
  --graph_connectivity_weight "${GRAPH_CONNECTIVITY_WEIGHT}"
  --question_overlap_weight "${QUESTION_OVERLAP_WEIGHT}"
  --evidence_sentences 16
  --evidence_per_doc 2
  --evidence_min_len 20
  --evidence_max_chars 1200
)

"${PYTHON_BIN}" "${ROOT_DIR}/rag_retrieve.py" "${common_args[@]}" \
  --raw_file "${ROOT_DIR}/data/hotpot_dev_distractor_v1.json" \
  --candidate_mode provided_context \
  --out_file "${ROOT_DIR}/rag_cache/distractor_top8_rebuilt.jsonl" \
  --metrics_file "${ROOT_DIR}/rag_cache/distractor_top8_rebuilt_metrics.json"

"${PYTHON_BIN}" "${ROOT_DIR}/rag_retrieve.py" "${common_args[@]}" \
  --raw_file "${ROOT_DIR}/data/hotpot_dev_fullwiki_v1.json" \
  --candidate_mode global_dense \
  --out_file "${ROOT_DIR}/rag_cache/fullwiki_top8_rebuilt.jsonl" \
  --metrics_file "${ROOT_DIR}/rag_cache/fullwiki_top8_rebuilt_metrics.json"

echo "[note] Paper numbers use artifacts/retrieval/*.jsonl, the frozen traces from the reported runs."
