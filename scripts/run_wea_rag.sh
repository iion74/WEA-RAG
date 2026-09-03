#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LIMIT="${LIMIT:-}"
LIVE_WEB="${LIVE_WEB:-0}"
source "${ROOT_DIR}/configs/paper.env"

mkdir -p "${ROOT_DIR}/predictions" "${ROOT_DIR}/metrics" "${ROOT_DIR}/logs"
SYSTEM_PROMPT="$(<"${ROOT_DIR}/configs/qa_short_strict.txt")"

if [[ "${LIVE_WEB}" == "1" ]]; then
  WEB_CACHE_FILE="${WEB_CACHE_FILE:-${ROOT_DIR}/rag_cache/live_web_cache.jsonl}"
else
  WEB_CACHE_FILE="${WEB_CACHE_FILE:-${ROOT_DIR}/artifacts/web_cache/hotpot_shared_web_cache.jsonl}"
fi

resolve_model() {
  case "$1" in
    qwen) MODEL_ID="Qwen/Qwen3-4B-Instruct-2507" ;;
    llama) MODEL_ID="meta-llama/Llama-3.2-3B-Instruct" ;;
    minicpm) MODEL_ID="openbmb/MiniCPM3-4B" ;;
    *) echo "unknown model key: $1" >&2; exit 2 ;;
  esac
}

selector_margin() {
  case "$1:$2" in
    qwen:distractor) echo 0.20 ;;
    qwen:fullwiki) echo 0.08 ;;
    llama:distractor|llama:fullwiki) echo 0.20 ;;
    minicpm:distractor) echo 0.03 ;;
    minicpm:fullwiki) echo 0.20 ;;
    *) echo 0.15 ;;
  esac
}

for model in ${MODEL_KEYS}; do
  resolve_model "${model}"
  for split in ${SPLITS}; do
    data_file="${ROOT_DIR}/data/hotpot_dev_${split}_v1.jsonl"
    retrieval_file="${ROOT_DIR}/artifacts/retrieval/${split}_top8.jsonl"
    margin="$(selector_margin "${model}" "${split}")"
    optional_args=()
    [[ -n "${LIMIT}" ]] && optional_args+=(--limit "${LIMIT}")
    if [[ "${LIVE_WEB}" != "1" ]]; then
      optional_args+=(--web_cache_only)
    fi

    echo "[wea-rag] model=${model} split=${split} margin=${margin} live_web=${LIVE_WEB}"
    "${PYTHON_BIN}" "${ROOT_DIR}/rag_eval/wea_eval.py" \
      --data_file "${data_file}" \
      --retrieval_file "${retrieval_file}" \
      --corpus_file "${ROOT_DIR}/rag_corpus/docs.jsonl" \
      --pred_file "${ROOT_DIR}/predictions/${model}_wea_rag_${split}.jsonl" \
      --metrics_file "${ROOT_DIR}/metrics/${model}_wea_rag_${split}.json" \
      --model "${MODEL_ID}" \
      --context_k "${CONTEXT_K}" \
      --max_context_chars "${MAX_CONTEXT_CHARS}" \
      --max_web_chars "${MAX_WEB_CHARS}" \
      --max_total_chars "${MAX_TOTAL_CHARS}" \
      --max_new_tokens "${WEA_MAX_NEW_TOKENS}" \
      --web_top_k "${WEB_TOP_K}" \
      --coverage_threshold "${COVERAGE_THRESHOLD}" \
      --web_gate_mode heuristic \
      --web_min_missing_terms "${MIN_MISSING_TERMS}" \
      --web_force_for_time_sensitive \
      --web_query_strategy "${WEB_QUERY_STRATEGY}" \
      --web_try_all_queries_on_miss \
      --web_cache_file "${WEB_CACHE_FILE}" \
      --stop_on_web_error \
      --dual_answer_select \
      --dual_slot_refine \
      --wea_fusion_mode merge \
      --dual_select_margin "${margin}" \
      --dual_wea_min_score "${WEA_MIN_SCORE}" \
      --dual_wea_min_token_support "${WEA_MIN_TOKEN_SUPPORT}" \
      --dual_require_context_match \
      --dual_rag_lock_score "${RAG_LOCK_SCORE}" \
      --system_prompt "${SYSTEM_PROMPT}" \
      --seed "${SEED}" \
      --overwrite "${optional_args[@]}"
  done
done
