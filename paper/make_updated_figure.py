"""Build the paper's split, fusion, and history sensitivity figure from saved outputs."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    resolution = json.loads(
        (RESULTS / "final_analysis" / "mf_b_selection_resolution.json").read_text(
            encoding="utf-8"
        )
    )
    bias = json.loads(
        (RESULTS / "final_analysis" / "bias_ranking.json").read_text(
            encoding="utf-8"
        )
    )
    chrono_tests = {"Bias": bias["chronological"]["test"]["ndcg_at_10"]}
    random_tests = {"Bias": bias["random"]["test"]["ndcg_at_10"]}
    for split_name, target in (
        ("chronological", chrono_tests),
        ("random", random_tests),
    ):
        ranking = resolution["ranking"][split_name]
        target.update(
            {
                "Most Popular": ranking["Popular"]["test"]["ndcg_at_10"],
                "MF A": ranking["MF A RMSE"]["test"]["ndcg_at_10"],
                "MF B (RMSE)": ranking["MF B RMSE (grid min)"]["test"]["ndcg_at_10"],
                "MF B (NDCG)": ranking["MF B NDCG"]["test"]["ndcg_at_10"],
                "Genre": ranking["Genre"]["test"]["ndcg_at_10"],
            }
        )
    chrono_curve = read_csv(
        RESULTS / "final_analysis" / "chronological_mf_b_ndcg_fusion_validation.csv"
    )
    random_curve = read_csv(
        RESULTS / "final_analysis" / "random_mf_b_ndcg_fusion_validation.csv"
    )
    chrono_sweep = {
        float(row["alpha"]): float(row["validation_ndcg_at_10"])
        for row in chrono_curve
    }
    random_sweep = {
        float(row["alpha"]): float(row["validation_ndcg_at_10"])
        for row in random_curve
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.4,
            "axes.titlesize": 8.0,
            "legend.fontsize": 6.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.65), constrained_layout=True)
    colors = {"chronological": "#345B72", "random": "#D36B3F"}
    model_keys = ["Bias", "Most Popular", "MF A", "MF B (RMSE)", "MF B (NDCG)", "Genre"]
    model_labels = [
        "Bias",
        "Most\nPopular",
        "MF A",
        "MF B\n(RMSE)",
        "MF B\n(NDCG)",
        "Genre",
    ]

    ax = axes[0]
    x = np.arange(len(model_keys))
    width = 0.34
    ax.bar(
        x - width / 2,
        [chrono_tests[model] for model in model_keys],
        width,
        label="Per-user chronological",
        color=colors["chronological"],
    )
    ax.bar(
        x + width / 2,
        [random_tests[model] for model in model_keys],
        width,
        label="Random per-user",
        color=colors["random"],
    )
    ax.set_xticks(x, model_labels)
    ax.set_xlabel("Model")
    ax.set_ylabel("Test NDCG@10")
    ax.set_ylim(0, 0.09)
    ax.set_title("A  Standalone test ranking", loc="left", fontweight="bold")
    ax.legend(
        frameon=False,
        loc="upper left",
        handlelength=1.1,
        labelspacing=0.25,
        labels=["Per-user chronological", "Random per-user"],
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E8EB", linewidth=0.55)
    ax.set_axisbelow(True)

    ax = axes[1]
    alphas = sorted(chrono_sweep)
    ax.plot(
        alphas,
        [chrono_sweep[alpha] for alpha in alphas],
        marker="o",
        markersize=2.7,
        linewidth=1.3,
        color=colors["chronological"],
        label="Per-user chronological",
    )
    ax.plot(
        alphas,
        [random_sweep[alpha] for alpha in alphas],
        marker="o",
        markersize=2.7,
        linewidth=1.3,
        color=colors["random"],
        label="Random per-user",
    )
    ax.scatter(
        [0.8, 1.0],
        [random_sweep[0.8], random_sweep[1.0]],
        s=23,
        facecolors="white",
        edgecolors=colors["random"],
        linewidths=1.1,
        zorder=4,
    )
    ax.annotate(
        r"$\alpha=0.8$",
        (0.8, random_sweep[0.8]),
        xytext=(-45, 5),
        textcoords="offset points",
        color=colors["random"],
        fontsize=6.2,
    )
    ax.annotate(
        r"$\alpha=1.0$",
        (1.0, random_sweep[1.0]),
        xytext=(-28, 5),
        textcoords="offset points",
        color=colors["random"],
        fontsize=6.2,
    )
    ax.set_xlabel(r"MF B weight, $\alpha$")
    ax.set_ylabel("Validation NDCG@10")
    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(0, 0.12)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_title("B  Validation z-score fusion", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left", handlelength=1.2, labelspacing=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E8EB", linewidth=0.55)
    ax.set_axisbelow(True)

    for ax in axes:
        ax.tick_params(length=2, pad=2)
        ax.margins(x=0.04)

    fig.savefig(PAPER / "split_fusion_history.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "split_fusion_history.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
