"""
Task 4: Comprehensive Model Benchmark.
Trains and compares all 5 models under a unified evaluation protocol:
1. Popularity (Bayesian Weighted Rating)
2. Content-Based (TF-IDF + Cosine Similarity)
3. Collaborative Filtering (SVD Matrix Factorization)
4. Standard Linear Hybrid (alpha = 0.6)
5. Proposed Calibrated & Long-Tail Aware Re-ranker (lambda = 0.5, gamma = 0.05)

Outputs standard ranking accuracy (P@10, R@10, F1@10, NDCG@10, MAP@10),
beyond-accuracy (Diversity, Novelty, Calibration D_KL), and exposure (Coverage, Gini).
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
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
    compute_map_at_k,
    compute_intra_list_diversity,
    compute_list_novelty,
    compute_aggregate_metrics
)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def run_benchmark(eval_users_limit=150, top_k=10):
    print("=" * 70)
    print("STARTING UNIFIED RECOMMENDER BENCHMARK")
    print(f"Top-K: {top_k}, User Cohort: {eval_users_limit} test users")
    print("=" * 70)

    # 1. Load Data
    data = load_dataset_pipeline()
    train_df = data["train_df"]
    test_df = data["test_df"]
    movies_df = data["movies_df"]
    all_genres = data["all_genres"]
    user_genre_profiles = data["user_genre_profiles"]
    trainset = data["trainset"]
    testset = data["testset"]

    total_catalog_movie_ids = movies_df["movieId"].unique().tolist()
    movie_genre_map = movies_df.set_index("movieId")["genre_list"].to_dict()

    # 2. Train Models
    print("\n[1/5] Fitting Popularity Model...")
    pop_model = PopularityRecommender().fit(train_df, movies_df)

    print("[2/5] Fitting Content-Based Model...")
    content_model = ContentBasedRecommender().fit(movies_df)

    print("[3/5] Fitting SVD Collaborative Filtering Model...")
    svd_model = SVDRecommender(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    svd_model.fit(trainset)

    # Compute rating prediction accuracy for SVD
    print("[SVD Evaluation] Testing rating prediction accuracy on per-user holdout...")
    svd_err = svd_model.evaluate_testset(testset)
    print(f"--> SVD Hold-out RMSE: {svd_err['rmse']:.4f}, MAE: {svd_err['mae']:.4f}")

    print("[4/5] Initializing Standard Hybrid Recommender (alpha=0.6)...")
    hybrid_model = HybridRecommender(pop_model, content_model, svd_model, alpha=0.6)

    print("[5/5] Initializing Calibrated & Long-Tail Re-ranker (lambda=0.5, gamma=0.05)...")
    calibrated_engine = CalibratedReranker(all_genres, lambda_param=0.5, gamma_param=0.05)

    # 3. Prepare User Evaluation Cohort
    test_user_groups = test_df.groupby("userId")
    eligible_users = []
    user_test_relevant = {}
    user_test_ratings = {}
    user_train_movies = train_df.groupby("userId")["movieId"].apply(set).to_dict()

    for user_id, group in test_user_groups:
        rel_movies = set(group[group["rating"] >= 4.0]["movieId"])
        if len(rel_movies) >= 1:
            eligible_users.append(user_id)
            user_test_relevant[user_id] = rel_movies
            user_test_ratings[user_id] = dict(zip(group["movieId"], group["rating"]))

    eval_cohort = eligible_users[:eval_users_limit]
    print(f"\nIdentified {len(eligible_users)} users with >= 1 relevant item (rating >= 4.0).")
    print(f"Benchmarking on {len(eval_cohort)} users across all 5 models...")

    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    models = [
        "Popularity (IMDb WR)",
        "Content-Based (TF-IDF)",
        "Collaborative (SVD)",
        "Linear Hybrid (alpha=0.6)",
        "Calibrated Reranked Hybrid"
    ]

    results = {m: {
        "precision": [], "recall": [], "f1": [], "ndcg": [], "map": [],
        "diversity": [], "novelty": [], "calibration_error": [],
        "all_recs": {}
    } for m in models}

    # Evaluate each user
    for user_id in tqdm(eval_cohort, desc="Benchmarking Users"):
        u_train_set = user_train_movies.get(user_id, set())
        u_relevant = user_test_relevant[user_id]
        u_test_dict = user_test_ratings[user_id]
        u_profile = user_genre_profiles[user_id]
        u_train_ratings = train_df[train_df["userId"] == user_id]

        candidate_ids = [m for m in total_catalog_movie_ids if m not in u_train_set]

        # 1. Popularity Recommendations
        pop_recs = pop_model.recommend(user_id, top_k=top_k, exclude_movie_ids=u_train_set)

        # 2. Content-Based Recommendations (Profile Centroid)
        content_recs = content_model.recommend(
            user_ratings_df=u_train_ratings, top_k=top_k, exclude_movie_ids=u_train_set
        )

        # 3. SVD Recommendations
        svd_recs = svd_model.recommend(
            user_id, total_catalog_movie_ids, top_k=top_k, exclude_movie_ids=u_train_set
        )

        # 4. Hybrid Recommendations
        hybrid_scores = hybrid_model.get_candidate_scores(
            user_id, u_train_ratings, seed_movie_id=None, candidate_movie_ids=candidate_ids, alpha=0.6
        )
        sorted_hybrid = sorted(hybrid_scores.items(), key=lambda kv: kv[1]["hybrid_score"], reverse=True)
        hybrid_recs = [m_id for m_id, _ in sorted_hybrid[:top_k]]

        # 5. Calibrated Re-ranker (Re-ranking top 100 hybrid candidates)
        top_candidates = [m_id for m_id, _ in sorted_hybrid[:100]]
        cand_score_dict = {m_id: hybrid_scores[m_id]["hybrid_score"] for m_id in top_candidates}
        calib_out = calibrated_engine.rerank(
            candidate_items=top_candidates,
            candidate_scores=cand_score_dict,
            user_genre_profile=u_profile,
            movie_genre_map=movie_genre_map,
            item_novelty=pop_model.item_novelty,
            top_k=top_k,
            lambda_param=0.5,
            gamma_param=0.05
        )
        calibrated_recs = calib_out["recommended_items"]

        rec_dict = {
            "Popularity (IMDb WR)": pop_recs,
            "Content-Based (TF-IDF)": content_recs,
            "Collaborative (SVD)": svd_recs,
            "Linear Hybrid (alpha=0.6)": hybrid_recs,
            "Calibrated Reranked Hybrid": calibrated_recs
        }

        # Calculate metrics for each model
        for m_name, rec_list in rec_dict.items():
            p, r, f = compute_precision_recall_f1_at_k(rec_list, u_relevant, k=top_k)
            ndcg = compute_ndcg_at_k(rec_list, u_test_dict, k=top_k)
            map_k = compute_map_at_k(rec_list, u_relevant, k=top_k)
            ild = compute_intra_list_diversity(rec_list, content_model.tfidf_matrix, content_model.movie_id_to_idx)
            nov = compute_list_novelty(rec_list, pop_model.item_novelty)

            # Compute calibration error for this recommendation list
            # Genre distribution of rec_list
            hypo_counts = np.zeros(calibrated_engine.num_genres, dtype=float)
            for m_id in rec_list:
                hypo_counts += calibrated_engine.compute_genre_vector(m_id, movie_genre_map)
            if np.sum(hypo_counts) > 0:
                q_dist = hypo_counts / np.sum(hypo_counts)
            else:
                q_dist = np.ones(calibrated_engine.num_genres) / calibrated_engine.num_genres
            kl_err = calibrated_engine.compute_kl_divergence(u_profile, q_dist)

            results[m_name]["precision"].append(p)
            results[m_name]["recall"].append(r)
            results[m_name]["f1"].append(f)
            results[m_name]["ndcg"].append(ndcg)
            results[m_name]["map"].append(map_k)
            results[m_name]["diversity"].append(ild)
            results[m_name]["novelty"].append(nov)
            results[m_name]["calibration_error"].append(kl_err)
            results[m_name]["all_recs"][user_id] = rec_list

    # Compute aggregate global metrics
    summary_rows = []
    for m_name in models:
        agg = compute_aggregate_metrics(
            results[m_name]["all_recs"], total_catalog_movie_ids, pop_model.item_novelty
        )
        row = {
            "Model": m_name,
            "Precision@10": np.mean(results[m_name]["precision"]),
            "Recall@10": np.mean(results[m_name]["recall"]),
            "F1@10": np.mean(results[m_name]["f1"]),
            "NDCG@10": np.mean(results[m_name]["ndcg"]),
            "MAP@10": np.mean(results[m_name]["map"]),
            "Diversity (ILD)": np.mean(results[m_name]["diversity"]),
            "Novelty (Bits)": np.mean(results[m_name]["novelty"]),
            "Calibration D_KL": np.mean(results[m_name]["calibration_error"]),
            "Catalog Coverage (%)": agg["catalog_coverage"],
            "Recommendation Gini": agg["recommendation_gini"],
            "Long-Tail Exposure (%)": agg["long_tail_ratio"]
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 100)
    print("UNIFIED BENCHMARK EVALUATION RESULTS (K=10)")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)

    # Save to CSV and JSON
    csv_path = os.path.join(REPORTS_DIR, "benchmark_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n[Benchmark] Saved summary table to: {csv_path}")

    # Serialize results
    json_path = os.path.join(REPORTS_DIR, "benchmark_metrics.json")
    with open(json_path, "w") as f:
        json.dump({
            "svd_rmse": float(svd_err["rmse"]),
            "svd_mae": float(svd_err["mae"]),
            "table": summary_df.to_dict(orient="records")
        }, f, indent=4)
    print(f"[Benchmark] Saved full metric data to: {json_path}")

    return summary_df, svd_err


if __name__ == "__main__":
    run_benchmark(eval_users_limit=100, top_k=10)
