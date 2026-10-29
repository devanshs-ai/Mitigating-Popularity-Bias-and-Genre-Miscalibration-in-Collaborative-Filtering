"""
Plots Figure 6: Pareto Frontier Comparison of Ranking Accuracy (NDCG@10) 
vs. Long-Tail Item Exposure (%).
Compares Proposed Novelty-Augmented Calibration (gamma sweep) against
the Popularity Penalty Baseline (beta sweep).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})


def plot_pareto():
    gamma_csv = os.path.join(REPORTS_DIR, "gamma_sweep.csv")
    pop_csv = os.path.join(REPORTS_DIR, "popularity_penalty_baseline.csv")

    df_gamma = pd.read_csv(gamma_csv)
    df_pop = pd.read_csv(pop_csv)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # 1. Proposed Method (Gamma sweep)
    ax.plot(
        df_gamma["Long_Tail_Ratio"],
        df_gamma["NDCG@10"],
        marker="o",
        color="#1f77b4",
        linewidth=2.2,
        label=r"Proposed Framework ($\gamma \in \{0, 0.01, 0.05, 0.1, 0.2\}$)"
    )

    # Annotate gamma points
    for _, row in df_gamma.iterrows():
        g_val = row["Gamma"]
        label_text = r"$\gamma$=" + f"{g_val:.2f}"
        offset = (6, 5) if g_val != 0.01 else (6, -12)
        ax.annotate(
            label_text,
            (row["Long_Tail_Ratio"], row["NDCG@10"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            color="#1f77b4",
            fontweight="bold"
        )

    # 2. Popularity Penalty Baseline (Beta sweep)
    ax.plot(
        df_pop["Long_Tail_Ratio"],
        df_pop["NDCG@10"],
        marker="s",
        color="#d62728",
        linewidth=2.2,
        linestyle="--",
        label=r"Popularity Penalty Baseline ($\beta \in \{0.05, 0.1, 0.2\}$)"
    )

    # Annotate beta points
    for _, row in df_pop.iterrows():
        b_val = row["Beta_Penalty"]
        label_text = r"$\beta$=" + f"{b_val:.2f}"
        ax.annotate(
            label_text,
            (row["Long_Tail_Ratio"], row["NDCG@10"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
            color="#d62728",
            fontweight="bold"
        )

    ax.set_xlabel("Long-Tail Item Exposure Ratio (%)")
    ax.set_ylabel("Ranking Accuracy (NDCG@10)")
    ax.set_title("Pareto Frontier: Accuracy vs. Long-Tail Discovery Trade-off")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)

    # Shading the superior region
    ax.fill_between(
        df_gamma["Long_Tail_Ratio"],
        df_gamma["NDCG@10"],
        alpha=0.08,
        color="#1f77b4"
    )

    plt.tight_layout()
    out_fig = os.path.join(FIGURES_DIR, "fig6_novelty_vs_penalty_pareto.png")
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"[Done] Saved Figure 6 to: {out_fig}")


if __name__ == "__main__":
    plot_pareto()
