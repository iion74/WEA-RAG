#!/usr/bin/env python3
"""Rebuild the paper result tables from frozen public artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util import exact_match_score
MODELS = {
    "qwen": "Qwen3-4B-Instruct-2507",
    "llama": "Llama-3.2-3B-Instruct",
    "minicpm": "MiniCPM3-4B",
}
SPLITS = ("distractor", "fullwiki")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path: Path):
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = row
    return rows


def retrieval_summary(path: Path):
    count = 0
    hit = recall = coverage = 0.0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            hit += float(row.get("hit", 0.0))
            recall += float(row.get("recall", 0.0))
            coverage += float(row.get("coverage", 0.0))
    return {
        "split": path.stem.replace("_top8", ""),
        "total": count,
        "hit_at_8": 100.0 * hit / count,
        "recall_at_8": 100.0 * recall / count,
        "full_coverage_at_8": 100.0 * coverage / count,
    }


def load_gold(split):
    path = ROOT / "data" / f"hotpot_dev_{split}_v1.jsonl"
    rows = load_predictions(path)
    return {qid: row.get("answer", "") for qid, row in rows.items()}


def transition_summary(rag_rows, wea_rows, gold):
    rag_correct = rag_wrong = rescue = harm = 0
    call_count = call_correct = call_rescue = call_harm = 0
    for qid, answer in gold.items():
        rag_ok = bool(exact_match_score(rag_rows[qid].get("prediction", ""), answer))
        wea_ok = bool(exact_match_score(wea_rows[qid].get("prediction", ""), answer))
        rag_correct += int(rag_ok)
        rag_wrong += int(not rag_ok)
        rescue += int((not rag_ok) and wea_ok)
        harm += int(rag_ok and (not wea_ok))
        if wea_rows[qid].get("queried_web", False):
            call_count += 1
            call_correct += int(wea_ok)
            call_rescue += int((not rag_ok) and wea_ok)
            call_harm += int(rag_ok and (not wea_ok))

    pct = lambda numerator, denominator: 100.0 * numerator / denominator if denominator else 0.0
    return {
        "rescue_count": rescue,
        "harm_count": harm,
        "net_gain_count": rescue - harm,
        "rescue_rate_over_rag_wrong": pct(rescue, rag_wrong),
        "harm_rate_over_rag_correct": pct(harm, rag_correct),
        "wea_activation_count": call_count,
        "wea_activation_rate": pct(call_count, len(gold)),
        "wea_final_correct_rate": pct(call_correct, call_count),
        "wea_rescue_rate": pct(call_rescue, call_count),
        "wea_harm_rate": pct(call_harm, call_count),
    }


def operational_summary(wea_rows):
    rows = list(wea_rows.values())
    query_count = sum(bool(row.get("queried_web", False)) for row in rows)
    used_count = sum(bool(row.get("used_web", False)) for row in rows)
    cache_hit_count = sum(bool(row.get("web_cache_hit", False)) for row in rows)
    total = len(rows)
    pct = lambda numerator, denominator: 100.0 * numerator / denominator if denominator else 0.0
    return {
        "web_used_rate": pct(used_count, total),
        "web_query_rate": pct(query_count, total),
        "cache_hit_rate": pct(cache_hit_count, query_count),
    }


def build_rows():
    rows = []
    for model_key, model_name in MODELS.items():
        for split in SPLITS:
            baseline = load_json(ARTIFACTS / "metrics" / f"{model_key}_baseline_{split}.json")
            rag = load_json(ARTIFACTS / "metrics" / f"{model_key}_rag_{split}.json")
            wea = load_json(ARTIFACTS / "metrics" / f"{model_key}_wea_rag_{split}.json")
            rag_rows = load_predictions(ARTIFACTS / "predictions" / f"{model_key}_rag_{split}.jsonl")
            wea_rows = load_predictions(ARTIFACTS / "predictions" / f"{model_key}_wea_rag_{split}.jsonl")
            gold = load_gold(split)
            total = int(wea["total"])
            transitions = transition_summary(rag_rows, wea_rows, gold)
            operations = operational_summary(wea_rows)
            row = {
                "model": model_name,
                "split": split,
                "total": total,
                "baseline_em": float(baseline["exact_match"]),
                "baseline_f1": float(baseline["f1"]),
                "rag_em": float(rag["exact_match"]),
                "rag_f1": float(rag["f1"]),
                "wea_rag_em": float(wea["exact_match"]),
                "wea_rag_f1": float(wea["f1"]),
                "delta_em": float(wea["exact_match"] - rag["exact_match"]),
                "delta_f1": float(wea["f1"] - rag["f1"]),
                "avg_latency_sec": float(wea.get("avg_latency_sec", 0.0)),
                **operations,
                **transitions,
            }
            rows.append(row)
    return rows


def write_markdown(rows, retrieval_rows, path: Path):
    lines = [
        "# Reproduced paper results",
        "",
        "| Model | Split | Baseline EM/F1 | RAG EM/F1 | WEA-RAG EM/F1 | Delta vs. RAG EM/F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['split']} | "
            f"{row['baseline_em']:.2f}/{row['baseline_f1']:.2f} | "
            f"{row['rag_em']:.2f}/{row['rag_f1']:.2f} | "
            f"{row['wea_rag_em']:.2f}/{row['wea_rag_f1']:.2f} | "
            f"{row['delta_em']:+.2f}/{row['delta_f1']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Retrieval quality",
            "",
            "| Split | Hit@8 | Recall@8 | Full coverage@8 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in retrieval_rows:
        lines.append(
            f"| {row['split']} | {row['hit_at_8']:.2f} | "
            f"{row['recall_at_8']:.2f} | {row['full_coverage_at_8']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    output_dir = ARTIFACTS / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    retrieval_rows = [
        retrieval_summary(ARTIFACTS / "retrieval" / f"{split}_top8.jsonl")
        for split in SPLITS
    ]
    payload = {"performance": rows, "retrieval": retrieval_rows}
    (output_dir / "paper_results_reproduced.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "paper_results_reproduced.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, retrieval_rows, output_dir / "paper_results_reproduced.md")
    print(output_dir / "paper_results_reproduced.md")


if __name__ == "__main__":
    main()
