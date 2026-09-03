#!/usr/bin/env python3
"""Validate the frozen artifacts before a public release."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODELS = ("qwen", "llama", "minicpm")
SPLITS = ("distractor", "fullwiki")
EXPECTED = {
    ("qwen", "distractor"): (47.86, 64.35, 53.44, 69.18, 54.00, 69.69),
    ("qwen", "fullwiki"): (27.60, 38.91, 44.19, 58.44, 45.32, 60.00),
    ("llama", "distractor"): (45.12, 58.21, 47.39, 60.70, 47.63, 60.88),
    ("llama", "fullwiki"): (28.66, 37.66, 38.77, 50.79, 39.08, 51.19),
    ("minicpm", "distractor"): (40.49, 54.85, 47.36, 61.15, 47.55, 61.38),
    ("minicpm", "fullwiki"): (28.09, 39.23, 39.85, 52.48, 40.61, 53.23),
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def assert_close(actual, expected, label):
    if abs(float(actual) - expected) > 0.015:
        raise AssertionError(f"{label}: expected {expected:.2f}, got {actual:.6f}")


def main():
    for model in MODELS:
        for split in SPLITS:
            baseline = load_json(ARTIFACTS / "metrics" / f"{model}_baseline_{split}.json")
            rag = load_json(ARTIFACTS / "metrics" / f"{model}_rag_{split}.json")
            wea = load_json(ARTIFACTS / "metrics" / f"{model}_wea_rag_{split}.json")
            expected = EXPECTED[(model, split)]
            actual = (
                baseline["exact_match"], baseline["f1"], rag["exact_match"],
                rag["f1"], wea["exact_match"], wea["f1"],
            )
            for index, (value, target) in enumerate(zip(actual, expected)):
                assert_close(value, target, f"{model}/{split}/metric-{index}")

            rag_rows = read_jsonl(ARTIFACTS / "predictions" / f"{model}_rag_{split}.jsonl")
            wea_rows = read_jsonl(ARTIFACTS / "predictions" / f"{model}_wea_rag_{split}.jsonl")
            if len(rag_rows) != 7405 or len(wea_rows) != 7405:
                raise AssertionError(f"{model}/{split}: expected 7,405 rows")
            if {str(row["id"]) for row in rag_rows} != {str(row["id"]) for row in wea_rows}:
                raise AssertionError(f"{model}/{split}: prediction IDs do not align")
            if any("dual_wea_pred" not in row for row in wea_rows):
                raise AssertionError(f"{model}/{split}: WEA trace field missing")
            if "dual_wea_min_score" not in wea:
                raise AssertionError(f"{model}/{split}: WEA metric field missing")

    for split in SPLITS:
        rows = read_jsonl(ARTIFACTS / "retrieval" / f"{split}_top8.jsonl")
        if len(rows) != 7405:
            raise AssertionError(f"{split}: expected 7,405 retrieval rows")

    cache_path = ARTIFACTS / "web_cache" / "hotpot_shared_web_cache.jsonl"
    cache_rows = read_jsonl(cache_path)
    if not cache_rows or any("query" not in row or "results" not in row for row in cache_rows):
        raise AssertionError("invalid web cache schema")
    lowered = cache_path.read_text(encoding="utf-8").lower()
    if "serpapi_api_key" in lowered or '"api_key"' in lowered:
        raise AssertionError("web cache appears to contain an API key field")

    print("validated 6 model/split result sets, 12 prediction traces, 2 retrieval traces, and the shared web cache")


if __name__ == "__main__":
    main()
