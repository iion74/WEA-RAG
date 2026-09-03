#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/wea-rag-matplotlib}"

"${PYTHON_BIN}" "${ROOT_DIR}/analysis/validate_artifacts.py"
"${PYTHON_BIN}" "${ROOT_DIR}/analysis/summarize_results.py"
"${PYTHON_BIN}" "${ROOT_DIR}/analysis/analyze_ablation.py" --bootstrap_reps 5000
"${PYTHON_BIN}" "${ROOT_DIR}/analysis/generate_figures.py"

echo "[done] reproduced tables and figures under artifacts/analysis and artifacts/figures/reproduced"
