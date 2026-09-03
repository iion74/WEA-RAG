#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

bash "${ROOT_DIR}/scripts/download_hotpotqa.sh"

for split in distractor fullwiki; do
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/convert_hotpot_json_to_jsonl.py" \
    --raw_file "${ROOT_DIR}/data/hotpot_dev_${split}_v1.json" \
    --out_file "${ROOT_DIR}/data/hotpot_dev_${split}_v1.jsonl"
done

echo "[done] HotpotQA inputs are ready under ${ROOT_DIR}/data"
