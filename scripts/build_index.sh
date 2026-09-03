#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"

"${PYTHON_BIN}" "${ROOT_DIR}/rag_build_index.py" \
  --corpus_file "${ROOT_DIR}/rag_corpus/docs.jsonl" \
  --index_dir "${ROOT_DIR}/rag_index" \
  --embed_model BAAI/bge-large-en-v1.5 \
  --batch_size 64 \
  --device "${DEVICE}"
