#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util import exact_match_score, f1_score


MODEL_MAP = {
    "qwen": ("qwen3-4b", "Qwen3-4B"),
    "llama": ("llama3.2-3b-instruct", "Llama-3.2-3B"),
    "minicpm": ("minicpm3-4b", "MiniCPM3-4B"),
}

SPLIT_FILE_MAP = {
    "dev_distractor": "hotpot_dev_distractor_v1.jsonl",
    "dev_fullwiki": "hotpot_dev_fullwiki_v1.jsonl",
}

SPLIT_SLUG = {
    "dev_distractor": "distractor",
    "dev_fullwiki": "fullwiki",
}


def load_gold(split: str) -> Dict[str, str]:
    path = ROOT / "data" / SPLIT_FILE_MAP[split]
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["id"])] = row.get("answer", "")
    return out


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def eval_preds(rows: List[Dict], gold: Dict[str, str]) -> Tuple[float, float, int]:
    em_sum = 0.0
    f1_sum = 0.0
    total = 0
    for row in rows:
        qid = str(row.get("id"))
        if qid not in gold:
            continue
        pred = row.get("prediction", "")
        ans = gold[qid]
        total += 1
        em_sum += exact_match_score(pred, ans)
        f1_sum += f1_score(pred, ans)[0]
    if total == 0:
        return 0.0, 0.0, 0
    return 100.0 * em_sum / total, 100.0 * f1_sum / total, total


def choose_answer(
    row: Dict,
    margin: float,
    wea_min_score: float,
    rag_lock_score: float,
) -> Tuple[str, str]:
    prediction = row.get("prediction", "")
    if not row.get("dual_select_applied", False):
        return prediction, "single"

    rag_pred = (row.get("dual_rag_pred") or "").strip()
    wea_pred = (row.get("dual_wea_pred") or "").strip()
    rag_score = float(row.get("dual_rag_score", -999.0))
    wea_score = float(row.get("dual_wea_score", -999.0))
    delta = wea_score - rag_score

    if not rag_pred and wea_pred:
        return wea_pred, "wea"
    if rag_pred and not wea_pred:
        return rag_pred, "rag"
    if not rag_pred and not wea_pred:
        return prediction, "single"

    if wea_score >= wea_min_score and delta >= margin:
        # keep a simple rag lock to avoid obviously confident rag being replaced
        if rag_score >= rag_lock_score and delta < (margin + 0.05):
            return rag_pred, "rag"
        return wea_pred, "wea"
    return rag_pred, "rag"


def apply_policy(rows: List[Dict], margin: float, wea_min_score: float, rag_lock_score: float) -> List[Dict]:
    out: List[Dict] = []
    for row in rows:
        new_row = dict(row)
        selected_pred, selected = choose_answer(row, margin, wea_min_score, rag_lock_score)
        new_row["prediction"] = selected_pred
        if row.get("dual_select_applied", False):
            new_row["dual_selected"] = selected
            new_row["dual_select_reason"] = "posthoc_tuned"
        new_row["posthoc_tuned"] = True
        new_row["posthoc_margin"] = margin
        new_row["posthoc_wea_min_score"] = wea_min_score
        new_row["posthoc_rag_lock_score"] = rag_lock_score
        out.append(new_row)
    return out


def count_dual(rows: List[Dict]) -> Tuple[int, int, int]:
    total = 0
    wea = 0
    rag = 0
    for row in rows:
        if not row.get("dual_select_applied", False):
            continue
        total += 1
        if row.get("dual_selected") == "wea":
            wea += 1
        else:
            rag += 1
    return total, wea, rag


def build_metrics_from_source(
    source_metrics: Dict,
    tuned_rows: List[Dict],
    em: float,
    f1: float,
    total: int,
    margin: float,
    wea_min_score: float,
    rag_lock_score: float,
) -> Dict:
    out = dict(source_metrics)
    out["exact_match"] = em
    out["f1"] = f1
    out["total"] = total
    out["dual_select_margin"] = margin
    out["dual_wea_min_score"] = wea_min_score
    out["dual_rag_lock_score"] = rag_lock_score
    dual_total, dual_wea, dual_rag = count_dual(tuned_rows)
    out["dual_select_total"] = dual_total
    out["dual_select_wea"] = dual_wea
    out["dual_select_rag"] = dual_rag
    out["dual_select_fused"] = 0
    out["posthoc_tuned"] = True
    return out


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen llama minicpm")
    parser.add_argument("--splits", default="dev_distractor dev_fullwiki")
    parser.add_argument("--margins", default="0.00,0.03,0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.22")
    parser.add_argument("--wea_min_scores", default="0.65,0.70,0.72,0.75,0.78,0.80")
    parser.add_argument("--rag_lock_score", type=float, default=0.88)
    parser.add_argument("--objective", default="em_f1_sum", choices=["em_f1_sum", "f1_only", "em_only"])
    parser.add_argument("--input_dir", default="artifacts/predictions")
    parser.add_argument("--metrics_dir", default="artifacts/metrics")
    parser.add_argument("--output_dir", default="artifacts/analysis/tuned_predictions")
    parser.add_argument("--report_json", default="artifacts/analysis/selector_tuning_reproduced.json")
    parser.add_argument("--report_md", default="artifacts/analysis/selector_tuning_reproduced.md")
    args = parser.parse_args()

    margins = parse_float_list(args.margins)
    wea_mins = parse_float_list(args.wea_min_scores)

    report_rows: List[Dict] = []

    for mk in args.models.split():
        if mk not in MODEL_MAP:
            continue
        _, model_label = MODEL_MAP[mk]
        for split in args.splits.split():
            if split not in SPLIT_FILE_MAP:
                continue

            slug = SPLIT_SLUG[split]
            source_pred = ROOT / args.input_dir / f"{mk}_wea_rag_{slug}.jsonl"
            source_metrics_path = ROOT / args.metrics_dir / f"{mk}_wea_rag_{slug}.json"
            target_pred = ROOT / args.output_dir / f"{mk}_wea_rag_{slug}.jsonl"
            target_metrics_path = ROOT / args.output_dir / f"{mk}_wea_rag_{slug}_metrics.json"

            if not source_pred.exists() or not source_metrics_path.exists():
                continue

            rows = load_jsonl(source_pred)
            source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
            gold = load_gold(split)

            best = None
            for margin in margins:
                for wea_min in wea_mins:
                    tuned_rows = apply_policy(rows, margin, wea_min, args.rag_lock_score)
                    em, f1, total = eval_preds(tuned_rows, gold)
                    if args.objective == "f1_only":
                        score = f1
                    elif args.objective == "em_only":
                        score = em
                    else:
                        score = em + f1
                    item = {
                        "margin": margin,
                        "wea_min_score": wea_min,
                        "em": em,
                        "f1": f1,
                        "score": score,
                        "total": total,
                        "rows": tuned_rows,
                    }
                    if best is None or item["score"] > best["score"]:
                        best = item

            if best is None:
                continue

            save_jsonl(target_pred, best["rows"])
            tuned_metrics = build_metrics_from_source(
                source_metrics=source_metrics,
                tuned_rows=best["rows"],
                em=best["em"],
                f1=best["f1"],
                total=best["total"],
                margin=best["margin"],
                wea_min_score=best["wea_min_score"],
                rag_lock_score=args.rag_lock_score,
            )
            target_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            target_metrics_path.write_text(
                json.dumps(tuned_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            report_rows.append(
                {
                    "model": model_label,
                    "split": split,
                    "source_trace": display_path(source_pred),
                    "target_trace": display_path(target_pred),
                    "source_em": float(source_metrics.get("exact_match", 0.0)),
                    "source_f1": float(source_metrics.get("f1", 0.0)),
                    "best_em": best["em"],
                    "best_f1": best["f1"],
                    "delta_em": best["em"] - float(source_metrics.get("exact_match", 0.0)),
                    "delta_f1": best["f1"] - float(source_metrics.get("f1", 0.0)),
                    "best_margin": best["margin"],
                    "best_wea_min_score": best["wea_min_score"],
                    "total_eval": best["total"],
                }
            )

    md_lines = []
    md_lines.append("# Post-hoc WEA-RAG Selector Tuning Report")
    md_lines.append("")
    md_lines.append(
        "| Model | Split | Source EM/F1 | Tuned EM/F1 | Delta EM/F1 | Best Margin | Best WEA Min Score |"
    )
    md_lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in report_rows:
        md_lines.append(
            "| {model} | {split} | {source_em:.2f}/{source_f1:.2f} | "
            "{best_em:.2f}/{best_f1:.2f} | {delta_em:+.2f}/{delta_f1:+.2f} | "
            "{best_margin:.2f} | {best_wea_min_score:.2f} |".format(**r)
        )
    md = "\n".join(md_lines) + "\n"
    print(md)

    if args.report_json:
        path = Path(args.report_json)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.report_md:
        path = Path(args.report_md)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
