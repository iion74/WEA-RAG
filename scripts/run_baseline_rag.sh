#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LIMIT="${LIMIT:-}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_RAG="${RUN_RAG:-1}"
source "${ROOT_DIR}/configs/paper.env"

mkdir -p "${ROOT_DIR}/predictions" "${ROOT_DIR}/metrics" "${ROOT_DIR}/logs"
SYSTEM_PROMPT="$(<"${ROOT_DIR}/configs/qa_short_strict.txt")"

resolve_model() {
  case "$1" in
    qwen)
      MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
      BASELINE_SCRIPT="basic_eval/qwen_eval.py"
      RAG_SCRIPT="rag_eval/qwen_rag_eval.py"
      ;;
    llama)
      MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"
      BASELINE_SCRIPT="basic_eval/llama_eval.py"
      RAG_SCRIPT="rag_eval/llama_rag_eval.py"
      ;;
    minicpm)
      MODEL_ID="openbmb/MiniCPM3-4B"
      BASELINE_SCRIPT="basic_eval/minicpm_eval.py"
      RAG_SCRIPT="rag_eval/minicpm_rag_eval.py"
      ;;
    *) echo "unknown model key: $1" >&2; exit 2 ;;
  esac
}

for model in ${MODEL_KEYS}; do
  resolve_model "${model}"
  for split in ${SPLITS}; do
    data_file="${ROOT_DIR}/data/hotpot_dev_${split}_v1.jsonl"
    retrieval_file="${ROOT_DIR}/artifacts/retrieval/${split}_top8.jsonl"
    limit_args=()
    [[ -n "${LIMIT}" ]] && limit_args=(--limit "${LIMIT}")

    if [[ "${RUN_BASELINE}" == "1" ]]; then
      echo "[baseline] model=${model} split=${split}"
      "${PYTHON_BIN}" "${ROOT_DIR}/${BASELINE_SCRIPT}" \
        --data_file "${data_file}" \
        --pred_file "${ROOT_DIR}/predictions/${model}_baseline_${split}.jsonl" \
        --metrics_file "${ROOT_DIR}/metrics/${model}_baseline_${split}.json" \
        --model "${MODEL_ID}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --max_context_chars "${MAX_CONTEXT_CHARS}" \
        --system_prompt "${SYSTEM_PROMPT}" \
        --seed "${SEED}" \
        --overwrite "${limit_args[@]}"
    fi

    if [[ "${RUN_RAG}" == "1" ]]; then
      echo "[rag] model=${model} split=${split}"
      "${PYTHON_BIN}" "${ROOT_DIR}/${RAG_SCRIPT}" \
        --data_file "${data_file}" \
        --retrieval_file "${retrieval_file}" \
        --corpus_file "${ROOT_DIR}/rag_corpus/docs.jsonl" \
        --pred_file "${ROOT_DIR}/predictions/${model}_rag_${split}.jsonl" \
        --metrics_file "${ROOT_DIR}/metrics/${model}_rag_${split}.json" \
        --model "${MODEL_ID}" \
        --context_k "${CONTEXT_K}" \
        --max_context_chars "${MAX_CONTEXT_CHARS}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --system_prompt "${SYSTEM_PROMPT}" \
        --seed "${SEED}" \
        --overwrite "${limit_args[@]}"
    fi
  done
done
