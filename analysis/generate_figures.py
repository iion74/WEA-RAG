#!/usr/bin/env python3
"""Generate the manuscript figures from frozen metrics and ablation outputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUT = ARTIFACTS / "figures" / "reproduced"
MODEL_KEYS = ("qwen", "llama", "minicpm")
MODEL_LABELS = ("Qwen", "Llama", "MiniCPM")
SPLITS = ("distractor", "fullwiki")
COLORS = {
    "baseline": "#5a5a5a",
    "rag": "#2878b5",
    "wea": "#2a9d68",
    "rescue": "#2a9d8f",
    "harm": "#e76f51",
    "net": "#264653",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric(model, method, split):
    return load_json(ARTIFACTS / "metrics" / f"{model}_{method}_{split}.json")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=240, bbox_inches="tight")
    plt.close(fig)


def annotate(ax, bars, fmt="{:.2f}"):
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for bar in bars:
        value = bar.get_height()
        offset = span * 0.018
        y = value + offset if value >= 0 else value - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(value),
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )


def grouped(ax, values, labels, colors, ylabel, title, zero=False):
    values = np.asarray(values, dtype=float)
    x = np.arange(values.shape[0])
    width = 0.78 / values.shape[1]
    for index, label in enumerate(labels):
        bars = ax.bar(
            x + (index - (values.shape[1] - 1) / 2) * width,
            values[:, index],
            width * 0.94,
            label=label,
            color=colors[index],
        )
        annotate(ax, bars)
    ax.set_xticks(x, MODEL_LABELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    if zero:
        ax.axhline(0, color="#666", linewidth=0.8)


def main_performance():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4))
    for row, split in enumerate(SPLITS):
        em, f1 = [], []
        for model in MODEL_KEYS:
            base, rag, wea = metric(model, "baseline", split), metric(model, "rag", split), metric(model, "wea_rag", split)
            em.append([base["exact_match"], rag["exact_match"], wea["exact_match"]])
            f1.append([base["f1"], rag["f1"], wea["f1"]])
        grouped(axes[row, 0], em, ("Baseline", "RAG", "WEA-RAG"), (COLORS["baseline"], COLORS["rag"], COLORS["wea"]), "EM", f"{split.title()} EM")
        grouped(axes[row, 1], f1, ("Baseline", "RAG", "WEA-RAG"), (COLORS["baseline"], COLORS["rag"], COLORS["wea"]), "F1", f"{split.title()} F1")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False)
    fig.suptitle("HotpotQA performance", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "main_performance.png")


def usage_figures(summary):
    indexed = {(row["model"], row["split"]): row for row in summary["performance"]}
    public_names = tuple(indexed_name for indexed_name in (
        "Qwen3-4B-Instruct-2507", "Llama-3.2-3B-Instruct", "MiniCPM3-4B"
    ))
    for split in SPLITS:
        values = []
        for model in public_names:
            data = indexed[(model, split)]
            values.append([
                data["wea_activation_rate"],
                data["web_used_rate"],
                data["web_query_rate"],
            ])
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        grouped(ax, values, ("WEA gate", "Web used", "Web query"), ("#457b9d", COLORS["wea"], "#e9c46a"), "Rate (%)", split.title())
        ax.legend(ncol=3, frameon=False, loc="upper center")
        fig.tight_layout()
        save(fig, f"wea_usage_{split}.png")


def quality_figures(summary):
    tuned = {(row["model"], row["split"]): row for row in summary["performance"]}
    model_names = ("Qwen3-4B-Instruct-2507", "Llama-3.2-3B-Instruct", "MiniCPM3-4B")
    for split in SPLITS:
        rows = [tuned[(model, split)] for model in model_names]
        counts = [[row["rescue_count"], row["harm_count"], row["net_gain_count"]] for row in rows]
        rates = [[row["rescue_rate_over_rag_wrong"], row["harm_rate_over_rag_correct"]] for row in rows]
        correct = [[row["wea_final_correct_rate"], row["wea_rescue_rate"], row["wea_harm_rate"]] for row in rows]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
        grouped(axes[0], counts, ("Rescue", "Harm", "Net"), (COLORS["rescue"], COLORS["harm"], COLORS["net"]), "Questions", "Outcome counts", zero=True)
        grouped(axes[1], rates, ("Rescue rate", "Harm rate"), (COLORS["rescue"], COLORS["harm"]), "Rate (%)", "WEA-case outcomes")
        grouped(axes[2], correct, ("Final correct", "Rescue", "Harm"), ("#2f4b7c", COLORS["rescue"], COLORS["harm"]), "Rate (%)", "Activated WEA cases")
        for ax in axes:
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(split.title(), fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save(fig, f"wea_quality_{split}.png")


def latency_figure():
    values = []
    labels = []
    for split in SPLITS:
        for model in MODEL_KEYS:
            values.append(metric(model, "wea_rag", split)["avg_latency_sec"])
            labels.append(f"{MODEL_LABELS[MODEL_KEYS.index(model)]}\n{split.title()}")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(np.arange(len(values)), values, color=[COLORS["rag"]] * 3 + [COLORS["wea"]] * 3)
    ax.set_xticks(np.arange(len(values)), labels)
    ax.set_ylabel("Seconds per question")
    ax.set_title("Average WEA-RAG latency", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    annotate(ax, bars)
    fig.tight_layout()
    save(fig, "latency.png")


def gate_ablation(ablation):
    policies = ("gate_t078_m2_time1", "gate_t078_m4_time1")
    names = ("Aggressive", "Conservative")
    rows = {(row["model"], row["split"], row["policy"]): row for row in ablation["gate_ablation"]}
    model_names = ("Qwen3-4B", "Llama-3.2-3B", "MiniCPM3-4B")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    xlabels, f1_values, call_values = [], [], []
    for split in ("dev_distractor", "dev_fullwiki"):
        for model, short in zip(model_names, MODEL_LABELS):
            xlabels.append(f"{short}\n{split.replace('dev_', '').title()}")
            f1_values.append([rows[(model, split, policy)]["delta_f1"] for policy in policies])
            call_values.append([rows[(model, split, policy)]["call_rate"] for policy in policies])
    for ax, values, ylabel, title in ((axes[0], f1_values, "F1 points", "F1 change vs. RAG"), (axes[1], call_values, "Rate (%)", "WEA activation rate")):
        values = np.asarray(values)
        x = np.arange(len(xlabels))
        width = 0.36
        for i, name in enumerate(names):
            bars = ax.bar(x + (i - 0.5) * width, values[:, i], width, label=name, color=("#e76f51", "#457b9d")[i])
            annotate(ax, bars)
        ax.set_xticks(x, xlabels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
        ax.axhline(0, color="#666", linewidth=0.8)
        ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "gate_ablation.png")


def selector_ablation(ablation):
    rows = defaultdict(dict)
    for row in ablation["selector_ablation"]:
        rows[(row["model"], row["split"])][row["policy"]] = row
    model_names = ("Qwen3-4B", "Llama-3.2-3B", "MiniCPM3-4B")
    xlabels, values = [], []
    for split in ("dev_distractor", "dev_fullwiki"):
        for model, short in zip(model_names, MODEL_LABELS):
            current = rows[(model, split)]
            tuned_f1 = current["stored_tuned"]["f1"]
            xlabels.append(f"{short}\n{split.replace('dev_', '').title()}")
            values.append([
                current["rag_keep"]["f1"] - tuned_f1,
                current["wea_force"]["f1"] - tuned_f1,
            ])
    values = np.asarray(values)
    fig, ax = plt.subplots(figsize=(10.5, 4.7))
    x = np.arange(len(xlabels))
    for i, (name, color) in enumerate((("Always keep RAG", COLORS["rag"]), ("Always take WEA", COLORS["harm"]))):
        bars = ax.bar(x + (i - 0.5) * 0.36, values[:, i], 0.36, label=name, color=color)
        annotate(ax, bars)
    ax.set_xticks(x, xlabels)
    ax.set_ylabel("F1 difference from tuned selector")
    ax.set_title("Selector ablation", fontweight="bold")
    ax.axhline(0, color="#555", linewidth=0.9)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "selector_ablation.png")


def retrieval_difficulty(ablation):
    order_c = ("high", "mid", "low")
    order_m = ("0-1", "2-3", "4+")
    grouped_rows = defaultdict(list)
    for row in ablation["retrieval_analysis"]:
        grouped_rows[(row["split"], row["coverage_bucket"], row["missing_bucket"])].append(row)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    for row_index, split in enumerate(("dev_distractor", "dev_fullwiki")):
        delta = np.full((3, 3), np.nan)
        rate = np.full((3, 3), np.nan)
        low_support = np.zeros((3, 3), dtype=bool)
        for i, coverage in enumerate(order_c):
            for j, missing in enumerate(order_m):
                rows = grouped_rows[(split, coverage, missing)]
                if not rows:
                    continue
                delta[i, j] = np.mean([item["delta_f1"] for item in rows])
                rate[i, j] = np.mean([item["call_rate"] for item in rows])
                low_support[i, j] = min(item["count"] for item in rows) < 20
        for column, (matrix, title, cmap) in enumerate(((delta, "Mean F1 change", "RdYlGn"), (rate, "Web query rate (%)", "Blues"))):
            ax = axes[row_index, column]
            image = ax.imshow(matrix, cmap=cmap, aspect="auto")
            for i in range(3):
                for j in range(3):
                    label = "low n" if low_support[i, j] else f"{matrix[i, j]:.2f}"
                    ax.text(j, i, label, ha="center", va="center", fontsize=9)
            ax.set_xticks(range(3), order_m)
            ax.set_yticks(range(3), order_c)
            ax.set_xlabel("Missing terms")
            ax.set_ylabel("Coverage")
            ax.set_title(f"{split.replace('dev_', '').title()}: {title}", fontweight="bold")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, "retrieval_difficulty.png")


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    ablation = load_json(ARTIFACTS / "analysis" / "ablation_reproduced.json")
    summary = load_json(ARTIFACTS / "analysis" / "paper_results_reproduced.json")
    main_performance()
    usage_figures(summary)
    quality_figures(summary)
    latency_figure()
    gate_ablation(ablation)
    selector_ablation(ablation)
    retrieval_difficulty(ablation)
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
