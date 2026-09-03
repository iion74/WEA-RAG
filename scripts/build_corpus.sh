#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${ROOT_DIR}/rag_build_corpus.py" \
  --train_file "${ROOT_DIR}/data/hotpot_train_v1.1.json" \
  --dev_files \
    "${ROOT_DIR}/data/hotpot_dev_distractor_v1.json" \
    "${ROOT_DIR}/data/hotpot_dev_fullwiki_v1.json" \
  --out_file "${ROOT_DIR}/rag_corpus/docs.jsonl" \
  --max_chars 2000

echo "[done] corpus written to ${ROOT_DIR}/rag_corpus/docs.jsonl"
