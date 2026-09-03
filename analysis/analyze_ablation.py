#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

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

GATE_POLICIES = [
    {
        "name": "rag_only",
        "label": "RAG only",
        "disable_all": True,
        "threshold": None,
        "min_missing": None,
        "force_time_sensitive": False,
    },
    {
        "name": "gate_t078_m2_time1",
        "label": "Gate t=0.78, m=2, time=on",
        "disable_all": False,
        "threshold": 0.78,
        "min_missing": 2,
        "force_time_sensitive": True,
    },
    {
        "name": "gate_t078_m2_time0",
        "label": "Gate t=0.78, m=2, time=off",
        "disable_all": False,
        "threshold": 0.78,
        "min_missing": 2,
        "force_time_sensitive": False,
    },
    {
        "name": "gate_t078_m4_time1",
        "label": "Gate t=0.78, m=4, time=on",
        "disable_all": False,
        "threshold": 0.78,
        "min_missing": 4,
        "force_time_sensitive": True,
    },
    {
        "name": "gate_t070_m4_time0",
        "label": "Gate t=0.70, m=4, time=off",
        "disable_all": False,
        "threshold": 0.70,
        "min_missing": 4,
        "force_time_sensitive": False,
    },
]

SELECTOR_POLICIES = [
    {
        "name": "stored_tuned",
        "label": "Stored tuned selector",
        "mode": "stored",
    },
    {
        "name": "rag_keep",
        "label": "Always keep RAG on dual cases",
        "mode": "rag_keep",
    },
    {
        "name": "wea_force",
        "label": "Always take WEA on dual cases",
        "mode": "wea_force",
    },
    {
        "name": "simple_margin_022_wea080_lock088",
        "label": "Simple selector m=0.22, WEA>=0.80, lock=0.88",
        "mode": "simple",
        "margin": 0.22,
        "wea_min_score": 0.80,
        "rag_lock_score": 0.88,
    },
    {
        "name": "simple_margin_000_wea065_lock088",
        "label": "Simple selector m=0.00, WEA>=0.65, lock=0.88",
        "mode": "simple",
        "margin": 0.00,
        "wea_min_score": 0.65,
        "rag_lock_score": 0.88,
    },
]

BUCKETS_COVERAGE = [
    ("low", lambda x: x < 0.70),
    ("mid", lambda x: 0.70 <= x < 0.85),
    ("high", lambda x: x >= 0.85),
]

BUCKETS_MISSING = [
    ("0-1", lambda x: x <= 1),
    ("2-3", lambda x: 2 <= x <= 3),
    ("4+", lambda x: x >= 4),
]


def pct(n: float, d: float) -> float:
    if d == 0:
        return 0.0
    return 100.0 * n / d


def load_jsonl_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_jsonl_map(path: Path) -> Dict[str, Dict]:
    return {str(row["id"]): row for row in load_jsonl_rows(path)}


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


def exact(pred: str, gold: str) -> int:
    return int(bool(exact_match_score(pred, gold)))


def f1_value(pred: str, gold: str) -> float:
    return float(f1_score(pred, gold)[0])


def metrics_from_predictions(pred_map: Dict[str, str], gold: Dict[str, str]) -> Dict[str, float]:
    ids = [qid for qid in gold if qid in pred_map]
    em = 0.0
    f1 = 0.0
    for qid in ids:
        pred = pred_map[qid]
        ans = gold[qid]
        em += exact(pred, ans)
        f1 += f1_value(pred, ans)
    total = len(ids)
    return {
        "total": total,
        "em": pct(em, total),
        "f1": pct(f1, total),
    }


def transition_metrics(ref_pred: Dict[str, str], target_pred: Dict[str, str], gold: Dict[str, str], call_ids: Iterable[str]) -> Dict[str, float]:
    call_ids = set(call_ids)
    total = 0
    ref_correct = 0
    ref_wrong = 0
    rescue = 0
    harm = 0
    call_count = 0
    call_correct = 0
    call_rescue = 0
    call_harm = 0

    for qid, ans in gold.items():
        if qid not in ref_pred or qid not in target_pred:
            continue
        total += 1
        r = exact(ref_pred[qid], ans)
        t = exact(target_pred[qid], ans)
        if r:
            ref_correct += 1
        else:
            ref_wrong += 1
        if (not r) and t:
            rescue += 1
        if r and (not t):
            harm += 1
        if qid in call_ids:
            call_count += 1
            if t:
                call_correct += 1
            if (not r) and t:
                call_rescue += 1
            if r and (not t):
                call_harm += 1

    return {
        "total_eval": total,
        "rescue_count": rescue,
        "harm_count": harm,
        "net_gain_count": rescue - harm,
        "rescue_rate_over_rag_wrong": pct(rescue, ref_wrong),
        "harm_rate_over_rag_correct": pct(harm, ref_correct),
        "call_count": call_count,
        "call_rate": pct(call_count, total),
        "call_correct_rate": pct(call_correct, call_count),
        "call_rescue_rate": pct(call_rescue, call_count),
        "call_harm_rate": pct(call_harm, call_count),
    }


def parse_reason_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


def gate_supports_row(row: Dict, threshold: float, min_missing: int, force_time_sensitive: bool) -> bool:
    reasons = parse_reason_list(row.get("web_gate_reason"))
    if "no_rag_context" in reasons:
        return True
    coverage = float(row.get("coverage", 1.0))
    missing_terms = row.get("missing_terms") or []
    low_coverage = coverage < threshold
    missing_heavy = len(missing_terms) >= min_missing
    time_sensitive = bool(row.get("time_sensitive_question", False))
    if low_coverage and missing_heavy:
        return True
    if force_time_sensitive and time_sensitive and (low_coverage or missing_heavy):
        return True
    return False


def choose_simple_selector(row: Dict, margin: float, wea_min_score: float, rag_lock_score: float) -> str:
    if not row.get("dual_select_applied", False):
        return row.get("prediction", "")

    rag_pred = (row.get("dual_rag_pred") or "").strip()
    wea_pred = (row.get("dual_wea_pred") or "").strip()
    rag_score = float(row.get("dual_rag_score", -999.0))
    wea_score = float(row.get("dual_wea_score", -999.0))
    delta = wea_score - rag_score

    if not rag_pred and wea_pred:
        return wea_pred
    if rag_pred and not wea_pred:
        return rag_pred
    if not rag_pred and not wea_pred:
        return row.get("prediction", "")
    if wea_score >= wea_min_score and delta >= margin:
        if rag_score >= rag_lock_score and delta < (margin + 0.05):
            return rag_pred
        return wea_pred
    return rag_pred


def apply_selector_policy(rows: List[Dict], policy: Dict) -> Dict[str, str]:
    pred_map: Dict[str, str] = {}
    for row in rows:
        qid = str(row["id"])
        if policy["mode"] == "stored":
            pred_map[qid] = row.get("prediction", "")
        elif policy["mode"] == "rag_keep":
            if row.get("dual_select_applied", False):
                pred_map[qid] = (row.get("dual_rag_pred") or "").strip() or row.get("prediction", "")
            else:
                pred_map[qid] = row.get("prediction", "")
        elif policy["mode"] == "wea_force":
            if row.get("dual_select_applied", False):
                pred_map[qid] = (row.get("dual_wea_pred") or "").strip() or (row.get("dual_rag_pred") or "").strip() or row.get("prediction", "")
            else:
                pred_map[qid] = row.get("prediction", "")
        elif policy["mode"] == "simple":
            pred_map[qid] = choose_simple_selector(
                row,
                margin=float(policy["margin"]),
                wea_min_score=float(policy["wea_min_score"]),
                rag_lock_score=float(policy["rag_lock_score"]),
            )
        else:
            raise ValueError(f"unknown selector mode: {policy['mode']}")
    return pred_map


def apply_gate_policy(rows_wea: List[Dict], rows_rag: Dict[str, Dict], policy: Dict) -> Tuple[Dict[str, str], Dict[str, float]]:
    pred_map: Dict[str, str] = {}
    call_ids: List[str] = []
    web_used = 0
    web_queries = 0
    unsupported_true = 0

    for row in rows_wea:
        qid = str(row["id"])
        rag_row = rows_rag[qid]
        gate_true = False
        if not policy["disable_all"]:
            gate_true = gate_supports_row(
                row,
                threshold=float(policy["threshold"]),
                min_missing=int(policy["min_missing"]),
                force_time_sensitive=bool(policy["force_time_sensitive"]),
            )

        if gate_true and bool(row.get("queried_web", False)):
            pred_map[qid] = row.get("prediction", "")
            call_ids.append(qid)
            if row.get("used_web", False):
                web_used += 1
            if row.get("queried_web", False):
                web_queries += 1
        else:
            pred_map[qid] = rag_row.get("prediction", "")
            if gate_true and not bool(row.get("queried_web", False)):
                unsupported_true += 1

    aux = {
        "web_used_count": web_used,
        "web_query_count": web_queries,
        "gate_true_no_query_count": unsupported_true,
        "call_ids": call_ids,
    }
    return pred_map, aux


def bucket_name_coverage(value: float) -> str:
    for name, fn in BUCKETS_COVERAGE:
        if fn(value):
            return name
    return "other"


def bucket_name_missing(value: int) -> str:
    for name, fn in BUCKETS_MISSING:
        if fn(value):
            return name
    return "other"


def build_retrieval_bucket_rows(model_label: str, split: str, rows_wea: List[Dict], rows_rag: Dict[str, Dict], gold: Dict[str, str]) -> List[Dict]:
    wea_by_id = {str(row["id"]): row for row in rows_wea}
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for row in rows_wea:
        qid = str(row["id"])
        coverage = float(row.get("coverage", 1.0))
        missing_count = len(row.get("missing_terms") or [])
        groups[(bucket_name_coverage(coverage), bucket_name_missing(missing_count))].append(qid)

    out: List[Dict] = []
    for (coverage_bucket, missing_bucket), ids in sorted(groups.items()):
        rag_pred = {qid: rows_rag[qid].get("prediction", "") for qid in ids}
        wea_pred = {qid: wea_by_id[qid].get("prediction", "") for qid in ids}

        em_rag = 0.0
        f1_rag = 0.0
        em_wea = 0.0
        f1_wea = 0.0
        rescue = 0
        harm = 0
        call_count = 0
        for qid in ids:
            ans = gold[qid]
            re = exact(rag_pred[qid], ans)
            rf = exact(wea_pred[qid], ans)
            em_rag += re
            em_wea += rf
            f1_rag += f1_value(rag_pred[qid], ans)
            f1_wea += f1_value(wea_pred[qid], ans)
            if (not re) and rf:
                rescue += 1
            if re and (not rf):
                harm += 1
            if bool(wea_by_id[qid].get("queried_web", False)):
                call_count += 1

        total = len(ids)
        out.append(
            {
                "model": model_label,
                "split": split,
                "coverage_bucket": coverage_bucket,
                "missing_bucket": missing_bucket,
                "count": total,
                "rag_em": pct(em_rag, total),
                "rag_f1": pct(f1_rag, total),
                "rag_wea_em": pct(em_wea, total),
                "rag_wea_f1": pct(f1_wea, total),
                "delta_em": pct(em_wea - em_rag, total),
                "delta_f1": pct(f1_wea - f1_rag, total),
                "rescue_count": rescue,
                "harm_count": harm,
                "net_gain_count": rescue - harm,
                "call_rate": pct(call_count, total),
            }
        )
    return out


def bootstrap_compare(
    rag_preds: Dict[str, Dict],
    wea_preds: Dict[str, Dict],
    gold: Dict[str, str],
    reps: int,
    seed: int,
    batch_size: int = 500,
) -> Dict[str, float]:
    ids = [qid for qid in gold if qid in rag_preds and qid in wea_preds]
    n = len(ids)
    rag_em = np.array([exact(rag_preds[qid].get("prediction", ""), gold[qid]) for qid in ids], dtype=np.float32)
    wea_em = np.array([exact(wea_preds[qid].get("prediction", ""), gold[qid]) for qid in ids], dtype=np.float32)
    rag_f1 = np.array([f1_value(rag_preds[qid].get("prediction", ""), gold[qid]) for qid in ids], dtype=np.float32)
    wea_f1 = np.array([f1_value(wea_preds[qid].get("prediction", ""), gold[qid]) for qid in ids], dtype=np.float32)

    rng = np.random.default_rng(seed)
    em_deltas: List[np.ndarray] = []
    f1_deltas: List[np.ndarray] = []

    done = 0
    while done < reps:
        cur = min(batch_size, reps - done)
        idx = rng.integers(0, n, size=(cur, n))
        em_delta = (wea_em[idx] - rag_em[idx]).mean(axis=1) * 100.0
        f1_delta = (wea_f1[idx] - rag_f1[idx]).mean(axis=1) * 100.0
        em_deltas.append(em_delta)
        f1_deltas.append(f1_delta)
        done += cur

    em_all = np.concatenate(em_deltas)
    f1_all = np.concatenate(f1_deltas)

    return {
        "n": n,
        "delta_em": float((wea_em.mean() - rag_em.mean()) * 100.0),
        "delta_f1": float((wea_f1.mean() - rag_f1.mean()) * 100.0),
        "em_ci_low": float(np.percentile(em_all, 2.5)),
        "em_ci_high": float(np.percentile(em_all, 97.5)),
        "f1_ci_low": float(np.percentile(f1_all, 2.5)),
        "f1_ci_high": float(np.percentile(f1_all, 97.5)),
        "em_p_le_zero": float((np.sum(em_all <= 0.0) + 1.0) / (len(em_all) + 1.0)),
        "f1_p_le_zero": float((np.sum(f1_all <= 0.0) + 1.0) / (len(f1_all) + 1.0)),
    }


def fmt(x: float | int | None, nd: int = 2) -> str:
    if x is None:
        return "-"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{nd}f}"


def build_markdown(report: Dict) -> str:
    lines: List[str] = []
    lines.append("# Cached WEA-RAG Paper Experiments")
    lines.append("")
    lines.append("## 1) Gate Ablation")
    lines.append("")
    lines.append("| Model | Split | Policy | EM | F1 | Delta EM vs RAG | Delta F1 vs RAG | Rescue | Harm | Net | CallRate | GateTrueNoQuery |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["gate_ablation"]:
        lines.append(
            "| {model} | {split} | {policy_label} | {em:.2f} | {f1:.2f} | {delta_em:+.2f} | {delta_f1:+.2f} | "
            "{rescue_count} | {harm_count} | {net_gain_count} | {call_rate:.2f} | {gate_true_no_query_count} |".format(**row)
        )

    lines.append("")
    lines.append("## 2) Selector Ablation")
    lines.append("")
    lines.append("| Model | Split | Policy | EM | F1 | Delta EM vs tuned | Delta F1 vs tuned | Rescue | Harm | Net |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["selector_ablation"]:
        lines.append(
            "| {model} | {split} | {policy_label} | {em:.2f} | {f1:.2f} | {delta_em:+.2f} | {delta_f1:+.2f} | "
            "{rescue_count} | {harm_count} | {net_gain_count} |".format(**row)
        )

    lines.append("")
    lines.append("## 3) Retrieval Difficulty Analysis")
    lines.append("")
    lines.append("| Model | Split | Coverage | Missing | Count | RAG EM | WEA-RAG EM | Delta EM | RAG F1 | WEA-RAG F1 | Delta F1 | WEA rate |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["retrieval_analysis"]:
        lines.append(
            "| {model} | {split} | {coverage_bucket} | {missing_bucket} | {count} | {rag_em:.2f} | {rag_wea_em:.2f} | "
            "{delta_em:+.2f} | {rag_f1:.2f} | {rag_wea_f1:.2f} | {delta_f1:+.2f} | {call_rate:.2f} |".format(**row)
        )

    lines.append("")
    lines.append("## 4) Bootstrap Significance")
    lines.append("")
    lines.append("| Model | Split | Delta EM | EM 95% CI | p(<=0) | Delta F1 | F1 95% CI | p(<=0) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in report["significance"]:
        lines.append(
            "| {model} | {split} | {delta_em:.2f} | [{em_ci_low:.2f}, {em_ci_high:.2f}] | {em_p_le_zero:.4f} | "
            "{delta_f1:.2f} | [{f1_ci_low:.2f}, {f1_ci_high:.2f}] | {f1_p_le_zero:.4f} |".format(**row)
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Gate ablation is an offline counterfactual over the cached traced run and is valid only for policies nested within the observed trigger set.")
    lines.append("- GateTrueNoQuery means the gate would fire under that policy, but the stored run did not issue a web query for that sample; those cases fall back to the RAG answer.")
    lines.append("- Selector ablation reuses stored dual predictions and scores from cached final prediction files.")
    lines.append("- Bootstrap significance compares RAG with stored tuned WEA-RAG over the same question set.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen llama minicpm")
    parser.add_argument("--splits", default="dev_distractor dev_fullwiki")
    parser.add_argument("--bootstrap_reps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out_json", default="artifacts/analysis/ablation_reproduced.json")
    parser.add_argument("--out_md", default="artifacts/analysis/ablation_reproduced.md")
    args = parser.parse_args()

    gate_rows: List[Dict] = []
    selector_rows: List[Dict] = []
    retrieval_rows: List[Dict] = []
    significance_rows: List[Dict] = []

    for mk in args.models.split():
        file_key, model_label = MODEL_MAP[mk]
        for split in args.splits.split():
            gold = load_gold(split)
            slug = SPLIT_SLUG[split]
            rag_path = ROOT / "artifacts" / "predictions" / f"{mk}_rag_{slug}.jsonl"
            wea_path = ROOT / "artifacts" / "predictions" / f"{mk}_wea_rag_{slug}.jsonl"
            rag_rows = load_jsonl_map(rag_path)
            wea_rows_list = load_jsonl_rows(wea_path)
            wea_rows_map = {str(row["id"]): row for row in wea_rows_list}

            base_rag_pred = {qid: row.get("prediction", "") for qid, row in rag_rows.items()}
            base_wea_pred = {qid: row.get("prediction", "") for qid, row in wea_rows_map.items()}
            rag_metrics = metrics_from_predictions(base_rag_pred, gold)
            base_metrics = metrics_from_predictions(base_wea_pred, gold)

            for policy in GATE_POLICIES:
                pred_map, gate_aux = apply_gate_policy(wea_rows_list, rag_rows, policy)
                metrics = metrics_from_predictions(pred_map, gold)
                trans = transition_metrics(base_rag_pred, pred_map, gold, gate_aux["call_ids"])
                gate_rows.append(
                    {
                        "model": model_label,
                        "split": split,
                        "policy": policy["name"],
                        "policy_label": policy["label"],
                        "em": metrics["em"],
                        "f1": metrics["f1"],
                        "delta_em": metrics["em"] - rag_metrics["em"],
                        "delta_f1": metrics["f1"] - rag_metrics["f1"],
                        "web_used_count": gate_aux["web_used_count"],
                        "web_query_count": gate_aux["web_query_count"],
                        "gate_true_no_query_count": gate_aux["gate_true_no_query_count"],
                        **trans,
                    }
                )

            for policy in SELECTOR_POLICIES:
                pred_map = apply_selector_policy(wea_rows_list, policy)
                metrics = metrics_from_predictions(pred_map, gold)
                dual_call_ids = [str(row["id"]) for row in wea_rows_list if row.get("dual_select_applied", False)]
                trans = transition_metrics(base_rag_pred, pred_map, gold, dual_call_ids)
                selector_rows.append(
                    {
                        "model": model_label,
                        "split": split,
                        "policy": policy["name"],
                        "policy_label": policy["label"],
                        "em": metrics["em"],
                        "f1": metrics["f1"],
                        "delta_em": metrics["em"] - base_metrics["em"],
                        "delta_f1": metrics["f1"] - base_metrics["f1"],
                        **trans,
                    }
                )

            retrieval_rows.extend(
                build_retrieval_bucket_rows(model_label, split, wea_rows_list, rag_rows, gold)
            )

            sig = bootstrap_compare(
                rag_preds=rag_rows,
                wea_preds=wea_rows_map,
                gold=gold,
                reps=args.bootstrap_reps,
                seed=args.seed,
            )
            significance_rows.append(
                {
                    "model": model_label,
                    "split": split,
                    **sig,
                }
            )

    report = {
        "config": {
            "models": args.models,
            "splits": args.splits,
            "rag_source": "artifacts/predictions/*_rag_*.jsonl",
            "wea_rag_source": "artifacts/predictions/*_wea_rag_*.jsonl",
            "bootstrap_reps": args.bootstrap_reps,
            "seed": args.seed,
        },
        "gate_ablation": gate_rows,
        "selector_ablation": selector_rows,
        "retrieval_analysis": retrieval_rows,
        "significance": significance_rows,
    }

    out_json = ROOT / args.out_json
    out_md = ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(report), encoding="utf-8")

    print(str(out_json))
    print(str(out_md))


if __name__ == "__main__":
    main()
