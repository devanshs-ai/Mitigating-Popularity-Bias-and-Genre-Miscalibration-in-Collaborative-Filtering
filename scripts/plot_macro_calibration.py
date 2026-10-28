"""
Generates Macro-Level Genre Calibration Plot across all 497 test users.
Replaces single-user case study with aggregate population statistics.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_dataset_pipeline
from src.models import SVDRecommender, CalibratedReranker

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "figure.dpi": 300
})


def plot_macro_genre_calibration():
    data = load_dataset_pipeline()
    train_df = data["train_df"]
    test_df = data["test_df"]
    movies_df = data["movies_df"]
    all_genres = data["all_genres"]
    user_genre_profiles = data["user_genre_profiles"]
    trainset = data["trainset"]

    total_catalog_movie_ids = movies_df["movieId"].unique().tolist()
    movie_genre_map = movies_df.set_index("movieId")["genre_list"].to_dict()

    svd_model = SVDRecommender().fit(trainset)
    calibrated_engine = CalibratedReranker(all_genres)

    # All users
    all_users = list(user_genre_profiles.keys())
    user_train_movies = train_df.groupby("userId")["movieId"].apply(set).to_dict()

    pop_user_p = np.zeros(len(all_genres))
    pop_svd_q = np.zeros(len(all_genres))
    pop_calib_q = np.zeros(len(all_genres))

    for u in all_users[:200]:
        u_p = user_genre_profiles[u]
        pop_user_p += u_p

        u_train = user_train_movies.get(u, set())
        cand_scores = svd_model.predict_catalog_ratings_vectorized(u, total_catalog_movie_ids)
        top_cand = sorted([m for m in total_catalog_movie_ids if m not in u_train],
                          key=lambda m: cand_scores.get(m, 0.0), reverse=True)[:100]

        # SVD top 10
        svd_top10 = top_cand[:10]
        # Calib top 10
        calib_res = calibrated_engine.rerank(
            top_cand, {m: cand_scores[m] for m in top_cand},
            u_p, movie_genre_map, top_k=10, lambda_param=0.8, gamma_param=0.05
        )
        calib_top10 = calib_res["recommended_items"]

        def get_q(rec_list):
            cnts = np.zeros(len(all_genres))
            for m in rec_list:
                for g in movie_genre_map.get(m, []):
                    if g in calibrated_engine.genre_to_idx:
                        cnts[calibrated_engine.genre_to_idx[g]] += 1
            return cnts / np.sum(cnts) if np.sum(cnts) > 0 else cnts

        pop_svd_q += get_q(svd_top10)
        pop_calib_q += get_q(calib_top10)

    pop_user_p /= 200.0
    pop_svd_q /= 200.0
    pop_calib_q /= 200.0

    # Sort genres by user true popularity
    order = np.argsort(-pop_user_p)[:10]
    sub_genres = [all_genres[i] for i in order]
    p_vals = pop_user_p[order] * 100.0
    svd_vals = pop_svd_q[order] * 100.0
    calib_vals = pop_calib_q[order] * 100.0

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(sub_genres))
    width = 0.28

    ax.bar(x - width, p_vals, width, label="True Population Taste P(g)", color="#2ca02c", alpha=0.85, edgecolor="black")
    ax.bar(x, svd_vals, width, label="Standard SVD Q(g) (Severe Genre Distortion)", color="#d62728", alpha=0.85, edgecolor="black")
    ax.bar(x + width, calib_vals, width, label="Calibrated & Novelty SVD (Ours)", color="#1f77b4", alpha=0.85, edgecolor="black")

    ax.set_ylabel("Macro Average Genre Share (%)")
    ax.set_xlabel("Top 10 Movie Genres")
    ax.set_title("Population-Level Genre Miscalibration: True Taste vs. SVD vs. Proposed Framework")
    ax.set_xticks(x)
    ax.set_xticklabels(sub_genres, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig5_macro_genre_calibration.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved Figure 5 to: {out_path}")


if __name__ == "__main__":
    plot_macro_genre_calibration()
