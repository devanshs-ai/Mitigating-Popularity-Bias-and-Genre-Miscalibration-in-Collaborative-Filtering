"""
Task 5: Calibration Trade-Off Experiment & Publication Figures.
Performs parametric sweep over calibration strength lambda in [0.0, 1.0].
Quantifies the Pareto frontier between ranking accuracy (NDCG@10) 
and taste calibration (D_KL) / catalog diversity.
Generates Figures 3, 4, and 5 for the academic paper.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_dataset_pipeline
from src.models import (
    PopularityRecommender,
    ContentBasedRecommender,
    SVDRecommender,
    HybridRecommender,
    CalibratedReranker
)
from src.evaluation import (
    compute_precision_recall_f1_at_k,
    compute_ndcg_at_k,
    compute_intra_list_diversity,
    compute_list_novelty,
    compute_aggregate_metrics
)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Publication styling
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


def run_calibration_experiment(eval_users_limit=100, top_k=10):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=" * 70)
    print("STARTING CALIBRATION PARAMETRIC SWEEP (LAMBDA 0.0 -> 1.0)")
    print("=" * 70)

    data = load_dataset_pipeline()
    train_df = data["train_df"]
    test_df = data["test_df"]
    movies_df = data["movies_df"]
    all_genres = data["all_genres"]
    user_genre_profiles = data["user_genre_profiles"]
    trainset = data["trainset"]

    total_catalog_movie_ids = movies_df["movieId"].unique().tolist()
    movie_genre_map = movies_df.set_index("movieId")["genre_list"].to_dict()

    pop_model = PopularityRecommender().fit(train_df, movies_df)
    content_model = ContentBasedRecommender().fit(movies_df)
    svd_model = SVDRecommender(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42).fit(trainset)
    hybrid_model = HybridRecommender(pop_model, content_model, svd_model, alpha=0.6)
    calibrated_engine = CalibratedReranker(all_genres)

    # Eligible test users
    eligible_users = []
    user_test_relevant = {}
    user_test_ratings = {}
    user_train_movies = train_df.groupby("userId")["movieId"].apply(set).to_dict()

    for user_id, group in test_df.groupby("userId"):
        rel_movies = set(group[group["rating"] >= 4.0]["movieId"])
        if len(rel_movies) >= 1:
            eligible_users.append(user_id)
            user_test_relevant[user_id] = rel_movies
            user_test_ratings[user_id] = dict(zip(group["movieId"], group["rating"]))

    eval_cohort = eligible_users[:eval_users_limit]

    # Precompute hybrid scores for top 100 candidates per user
    print(f"\nPrecomputing candidate pools for {len(eval_cohort)} test users...")
    user_candidates = {}
    for user_id in eval_cohort:
        u_train_set = user_train_movies.get(user_id, set())
        u_train_ratings = train_df[train_df["userId"] == user_id]
        cand_ids = [m for m in total_catalog_movie_ids if m not in u_train_set]
        h_scores = hybrid_model.get_candidate_scores(user_id, u_train_ratings, None, cand_ids, alpha=0.6)
        sorted_h = sorted(h_scores.items(), key=lambda kv: kv[1]["hybrid_score"], reverse=True)[:100]
        top_ids = [m for m, _ in sorted_h]
        user_candidates[user_id] = {
            "top_ids": top_ids,
            "score_dict": {m: h_scores[m]["hybrid_score"] for m in top_ids}
        }

    lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sweep_results = []

    for lam in tqdm(lambda_values, desc="Sweeping Lambda"):
        lam_prec = []
        lam_ndcg = []
        lam_div = []
        lam_nov = []
        lam_kl = []
        all_lam_recs = {}

        for user_id in eval_cohort:
            u_relevant = user_test_relevant[user_id]
            u_test_dict = user_test_ratings[user_id]
            u_profile = user_genre_profiles[user_id]
            top_ids = user_candidates[user_id]["top_ids"]
            score_dict = user_candidates[user_id]["score_dict"]

            res = calibrated_engine.rerank(
                candidate_items=top_ids,
                candidate_scores=score_dict,
                user_genre_profile=u_profile,
                movie_genre_map=movie_genre_map,
                item_novelty=pop_model.item_novelty,
                top_k=top_k,
                lambda_param=lam,
                gamma_param=0.05
            )

            rec_list = res["recommended_items"]
            all_lam_recs[user_id] = rec_list

            p, _, _ = compute_precision_recall_f1_at_k(rec_list, u_relevant, k=top_k)
            ndcg = compute_ndcg_at_k(rec_list, u_test_dict, k=top_k)
            ild = compute_intra_list_diversity(rec_list, content_model.tfidf_matrix, content_model.movie_id_to_idx)
            nov = compute_list_novelty(rec_list, pop_model.item_novelty)

            lam_prec.append(p)
            lam_ndcg.append(ndcg)
            lam_div.append(ild)
            lam_nov.append(nov)
            lam_kl.append(res["calibration_error"])

        agg = compute_aggregate_metrics(all_lam_recs, total_catalog_movie_ids, pop_model.item_novelty)
        sweep_results.append({
            "lambda": lam,
            "Precision@10": np.mean(lam_prec),
            "NDCG@10": np.mean(lam_ndcg),
            "Diversity_ILD": np.mean(lam_div),
            "Novelty_Bits": np.mean(lam_nov),
            "Calibration_D_KL": np.mean(lam_kl),
            "Catalog_Coverage_Pct": agg["catalog_coverage"],
            "Gini_Exposure": agg["recommendation_gini"],
            "Long_Tail_Ratio": agg["long_tail_ratio"]
        })

    sweep_df = pd.DataFrame(sweep_results)
    print("\n" + "=" * 90)
    print("CALIBRATION SWEEP RESULTS (LAMBDA vs ACCURACY vs BIAS)")
    print("=" * 90)
    print(sweep_df.to_string(index=False))
    print("=" * 90)

    sweep_csv_path = os.path.join(REPORTS_DIR, "calibration_sweep_results.csv")
    sweep_df.to_csv(sweep_csv_path, index=False)

    # -------------------------------------------------------------
    # Figure 3: Pareto Frontier Trade-Off (NDCG@10 vs D_KL and Diversity)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Ax1: NDCG@10 and Calibration Error vs Lambda
    ax1.plot(sweep_df["lambda"], sweep_df["NDCG@10"], marker="o", color="#1f77b4", linewidth=2.2, label="NDCG@10 (Ranking Accuracy)")
    ax1.set_xlabel(r"Calibration Parameter $\lambda$ (0 = Accuracy-only, 1 = Full Calibration)")
    ax1.set_ylabel("NDCG@10 Score", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(sweep_df["lambda"], sweep_df["Calibration_D_KL"], marker="s", color="#d62728", linewidth=2.2, linestyle="--", label="Calibration Error $D_{KL}$ (Taste Distortion)")
    ax1_twin.set_ylabel(r"Mean $D_{KL}(P \parallel Q)$ (Lower is Better)", color="#d62728")
    ax1_twin.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title(r"(a) Accuracy vs. Taste Distortion ($D_{KL}$)")

    # Ax2: Pareto Frontier - NDCG@10 vs Intra-List Diversity
    sc = ax2.scatter(sweep_df["Calibration_D_KL"], sweep_df["NDCG@10"], c=sweep_df["lambda"], cmap="viridis", s=90, edgecolors="black", zorder=5)
    ax2.plot(sweep_df["Calibration_D_KL"], sweep_df["NDCG@10"], color="gray", linestyle=":", alpha=0.7)
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label(r"$\lambda$ Value")

    for i, row in sweep_df.iterrows():
        if row["lambda"] in [0.0, 0.2, 0.5, 0.8, 1.0]:
            ax2.annotate(f"λ={row['lambda']:.1f}", (row["Calibration_D_KL"], row["NDCG@10"]),
                         xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax2.set_xlabel(r"Calibration Error $D_{KL}$ (Lower is Better)")
    ax2.set_ylabel("NDCG@10")
    ax2.set_title("(b) Pareto Frontier: Accuracy vs. Calibration")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "fig3_pareto_accuracy_vs_calibration.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"[Experiment] Saved Figure 3 to: {fig3_path}")

    # -------------------------------------------------------------
    # Figure 4: Catalog Coverage and Gini Inequality
    # -------------------------------------------------------------
    benchmark_csv = os.path.join(REPORTS_DIR, "benchmark_summary.csv")
    if os.path.exists(benchmark_csv):
        b_df = pd.read_csv(benchmark_csv)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        model_labels = ["Popularity", "Content", "SVD", "Hybrid", "Calibrated (Ours)"]
        colors = ["#7f7f7f", "#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"]

        # Coverage bar plot
        ax1.bar(model_labels, b_df["Catalog Coverage (%)"], color=colors, edgecolor="black", alpha=0.85)
        ax1.set_ylabel("Catalog Coverage (%) - Higher is Better")
        ax1.set_title("(a) Aggregate Catalog Coverage Across Models")
        ax1.set_xticklabels(model_labels, rotation=25, ha="right")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        # Gini inequality bar plot
        ax2.bar(model_labels, b_df["Recommendation Gini"], color=colors, edgecolor="black", alpha=0.85)
        ax2.set_ylabel("Exposure Gini Index - Lower is Fairer")
        ax2.set_title("(b) Popularity Exposure Inequality (Gini)")
        ax2.set_xticklabels(model_labels, rotation=25, ha="right")
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig4_path = os.path.join(FIGURES_DIR, "fig4_exposure_and_coverage_comparison.png")
        fig.savefig(fig4_path, dpi=300)
        plt.close(fig)
        print(f"[Experiment] Saved Figure 4 to: {fig4_path}")

    # -------------------------------------------------------------
    # Figure 5: Case Study: Popping the Filter Bubble for User 1
    # -------------------------------------------------------------
    sample_user = eval_cohort[0]
    u_profile = user_genre_profiles[sample_user]

    # Get recs from uncalibrated SVD vs Calibrated Reranker
    u_train_set = user_train_movies.get(sample_user, set())
    svd_recs = svd_model.recommend(sample_user, total_catalog_movie_ids, top_k=top_k, exclude_movie_ids=u_train_set)

    calib_sample = calibrated_engine.rerank(
        candidate_items=user_candidates[sample_user]["top_ids"],
        candidate_scores=user_candidates[sample_user]["score_dict"],
        user_genre_profile=u_profile,
        movie_genre_map=movie_genre_map,
        item_novelty=pop_model.item_novelty,
        top_k=top_k,
        lambda_param=0.6,
        gamma_param=0.05
    )
    calib_recs = calib_sample["recommended_items"]

    # Compute genre distributions
    def get_genre_dist(rec_list):
        counts = np.zeros(len(all_genres))
        for m_id in rec_list:
            for g in movie_genre_map.get(m_id, []):
                if g in calibrated_engine.genre_to_idx:
                    counts[calibrated_engine.genre_to_idx[g]] += 1
        return counts / np.sum(counts) if np.sum(counts) > 0 else counts

    svd_genre_dist = get_genre_dist(svd_recs)
    calib_genre_dist = get_genre_dist(calib_recs)

    # Focus on top 7 most prominent genres for user
    top_user_genre_indices = np.argsort(-u_profile)[:7]
    sub_genres = [all_genres[i] for i in top_user_genre_indices]
    u_sub = u_profile[top_user_genre_indices]
    svd_sub = svd_genre_dist[top_user_genre_indices]
    calib_sub = calib_genre_dist[top_user_genre_indices]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sub_genres))
    width = 0.25

    ax.bar(x - width, u_sub * 100, width, label="True Historical Preference P(g|u)", color="#2ca02c", alpha=0.85, edgecolor="black")
    ax.bar(x, svd_sub * 100, width, label="Standard SVD (Filter Bubble Distortion)", color="#ff7f0e", alpha=0.85, edgecolor="black")
    ax.bar(x + width, calib_sub * 100, width, label="Calibrated Re-ranker (Ours)", color="#1f77b4", alpha=0.85, edgecolor="black")

    ax.set_ylabel("Genre Proportion (%)")
    ax.set_xlabel("Movie Genre")
    ax.set_title(f"Case Study (User #{sample_user}): True Preference vs. SVD vs. Calibrated Re-ranking")
    ax.set_xticks(x)
    ax.set_xticklabels(sub_genres, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig5_path = os.path.join(FIGURES_DIR, "fig5_user_radar_taste_calibration.png")
    fig.savefig(fig5_path, dpi=300)
    plt.close(fig)
    print(f"[Experiment] Saved Figure 5 to: {fig5_path}")

    return sweep_df


if __name__ == "__main__":
    run_calibration_experiment(eval_users_limit=100, top_k=10)
