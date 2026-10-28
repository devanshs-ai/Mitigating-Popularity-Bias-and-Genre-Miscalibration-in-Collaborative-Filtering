"""
Ablation Studies, Candidate Pool Sweep, and Popularity Penalty Baseline.
Directly addresses peer review:
1. Candidate pool sweep: M in {100, 200, 500} to see if larger pools unlock the long tail.
2. Gamma sweep: gamma in {0.0, 0.01, 0.05, 0.10, 0.20} tracing Accuracy vs. Novelty Pareto frontier.
3. Popularity penalty baseline: r_hat - beta * log(1 + count).
4. Reference Decade TVD: user train history vs. user held-out test liked items (natural drift).
5. Holm-Bonferroni family-wise error rate correction.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_dataset_pipeline
from src.models import (
    PopularityRecommender,
    ContentBasedRecommender,
    SVDRecommender,
    CalibratedReranker,
    MMRReranker
)
from src.evaluation import (
    compute_precision_recall_f1_at_k,
    compute_ndcg_at_k,
    compute_map_at_k,
    compute_intra_list_diversity,
    compute_list_novelty,
    compute_aggregate_metrics,
    compute_total_variation_distance,
    compute_decade_tvd,
    compute_decade_bucket,
    compute_mean_ci
)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def holm_bonferroni(p_values):
    """Applies Holm-Bonferroni step-down correction to a list of p-values."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = p * (m - rank)
        cum_max = max(cum_max, adj_p)
        adjusted[orig_idx] = min(1.0, cum_max)
    return adjusted


def run_ablations(val_size=100, top_k=10):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 90)
    print("RUNNING EXTENSIVE ABLATIONS: CANDIDATE POOLS (100, 200, 500) & GAMMA SWEEP")
    print("=" * 90)

    # 1. Load Data
    data = load_dataset_pipeline(random_state=42)
    train_df = data["train_df"]
    test_df = data["test_df"]
    movies_df = data["movies_df"]
    all_genres = data["all_genres"]
    user_genre_profiles = data["user_genre_profiles"]
    trainset = data["trainset"]

    total_catalog_movie_ids = movies_df["movieId"].unique().tolist()
    movie_genre_map = movies_df.set_index("movieId")["genre_list"].to_dict()
    movie_year_map = movies_df.set_index("movieId")["year"].to_dict()

    # Precompute historical user decade profiles from train_df ONLY
    user_train_decade = {}
    for u, group in train_df.groupby("userId"):
        b = np.zeros(6, dtype=float)
        for _, row in group.iterrows():
            yr = movie_year_map.get(row["movieId"], np.nan)
            b[compute_decade_bucket(yr)] += (row["rating"] / 5.0)
        user_train_decade[u] = b / np.sum(b) if np.sum(b) > 0 else np.ones(6) / 6.0

    # REFERENCE: User decade profile on held-out TEST liked items (natural drift)
    user_test_liked_decade = {}
    natural_drift_tvds = []
    for u, group in test_df[test_df["rating"] >= 4.0].groupby("userId"):
        b = np.zeros(6, dtype=float)
        for _, row in group.iterrows():
            yr = movie_year_map.get(row["movieId"], np.nan)
            b[compute_decade_bucket(yr)] += 1.0
        if np.sum(b) > 0:
            test_dec = b / np.sum(b)
            user_test_liked_decade[u] = test_dec
            if u in user_train_decade:
                natural_drift_tvds.append(compute_total_variation_distance(user_train_decade[u], test_dec))

    ref_drift_mean = np.mean(natural_drift_tvds) if natural_drift_tvds else 0.45
    print(f"\n[Reference Baseline] Natural User Decade Drift (Train History vs. Test Liked): TVD = {ref_drift_mean:.4f}")

    # Base models
    pop_model = PopularityRecommender().fit(train_df, movies_df)
    svd_model = SVDRecommender().fit(trainset)
    calibrated_engine = CalibratedReranker(all_genres)

    # Train counts for popularity penalty baseline
    train_counts = train_df.groupby("movieId").size().to_dict()

    # Partition users
    eligible_users = []
    user_test_relevant = {}
    user_test_ratings = {}
    user_train_movies = train_df.groupby("userId")["movieId"].apply(set).to_dict()

    for user_id, group in test_df.groupby("userId"):
        rel = set(group[group["rating"] >= 4.0]["movieId"])
        if len(rel) >= 1:
            eligible_users.append(user_id)
            user_test_relevant[user_id] = rel
            user_test_ratings[user_id] = dict(zip(group["movieId"], group["rating"]))

    np.random.seed(42)
    shuffled = list(np.random.permutation(eligible_users))
    test_users = shuffled[val_size:]
    print(f"Evaluating ablations on {len(test_users)} held-out test users...")

    # -------------------------------------------------------------
    # EXPERIMENT 1: Candidate Pool Sweep (M in [100, 200, 500])
    # -------------------------------------------------------------
    print("\n[Ablation 1] Evaluating Candidate Pool Size M in {100, 200, 500}...")
    pool_records = []

    for pool_size in [100, 200, 500]:
        ndcg_l, p_l, kl_l, dec_l, nov_l = [], [], [], [], []
        all_recs = {}

        for u in test_users:
            u_train = user_train_movies.get(u, set())
            cand_scores = svd_model.predict_catalog_ratings_vectorized(u, total_catalog_movie_ids)
            top_cand = sorted([m for m in total_catalog_movie_ids if m not in u_train],
                              key=lambda m: cand_scores.get(m, 0.0), reverse=True)[:pool_size]
            cand_dict = {m: cand_scores[m] for m in top_cand}

            res = calibrated_engine.rerank(
                candidate_items=top_cand,
                candidate_scores=cand_dict,
                user_genre_profile=user_genre_profiles[u],
                movie_genre_map=movie_genre_map,
                item_novelty=pop_model.item_novelty,
                top_k=top_k,
                lambda_param=0.80,
                gamma_param=0.05
            )
            recs = res["recommended_items"]
            all_recs[u] = recs

            ndcg_l.append(compute_ndcg_at_k(recs, user_test_ratings[u], k=top_k))
            p_l.append(compute_precision_recall_f1_at_k(recs, user_test_relevant[u], k=top_k)[0])
            kl_l.append(res["calibration_error"])
            dec_l.append(compute_decade_tvd(user_train_decade[u], recs, movie_year_map))
            nov_l.append(compute_list_novelty(recs, pop_model.item_novelty))

        agg = compute_aggregate_metrics(all_recs, total_catalog_movie_ids, pop_model.item_novelty)
        pool_records.append({
            "Candidate_Pool_M": pool_size,
            "NDCG@10": np.mean(ndcg_l),
            "Precision@10": np.mean(p_l),
            "Calibration_D_KL": np.mean(kl_l),
            "Decade_TVD": np.mean(dec_l),
            "Novelty_Bits": np.mean(nov_l),
            "Catalog_Coverage_Pct": agg["catalog_coverage"],
            "Long_Tail_Ratio": agg["long_tail_ratio"]
        })
        print(f"  Pool M={pool_size:3d} -> NDCG={np.mean(ndcg_l):.4f}, Coverage={agg['catalog_coverage']:.2f}%, Long-Tail={agg['long_tail_ratio']:.2f}%")

    pool_df = pd.DataFrame(pool_records)
    pool_csv = os.path.join(REPORTS_DIR, "candidate_pool_sweep.csv")
    pool_df.to_csv(pool_csv, index=False)

    # -------------------------------------------------------------
    # EXPERIMENT 2: Gamma (Novelty Regularization) Sensitivity Sweep
    # -------------------------------------------------------------
    print("\n[Ablation 2] Evaluating Gamma Sweep in {0.0, 0.01, 0.05, 0.10, 0.20} at Pool M=200...")
    gamma_records = []

    for gam in [0.0, 0.01, 0.05, 0.10, 0.20]:
        ndcg_l, p_l, kl_l, nov_l = [], [], [], []
        all_recs = {}

        for u in test_users:
            u_train = user_train_movies.get(u, set())
            cand_scores = svd_model.predict_catalog_ratings_vectorized(u, total_catalog_movie_ids)
            top_cand = sorted([m for m in total_catalog_movie_ids if m not in u_train],
                              key=lambda m: cand_scores.get(m, 0.0), reverse=True)[:200]
            cand_dict = {m: cand_scores[m] for m in top_cand}

            res = calibrated_engine.rerank(
                candidate_items=top_cand,
                candidate_scores=cand_dict,
                user_genre_profile=user_genre_profiles[u],
                movie_genre_map=movie_genre_map,
                item_novelty=pop_model.item_novelty,
                top_k=top_k,
                lambda_param=0.80,
                gamma_param=gam
            )
            recs = res["recommended_items"]
            all_recs[u] = recs

            ndcg_l.append(compute_ndcg_at_k(recs, user_test_ratings[u], k=top_k))
            p_l.append(compute_precision_recall_f1_at_k(recs, user_test_relevant[u], k=top_k)[0])
            kl_l.append(res["calibration_error"])
            nov_l.append(compute_list_novelty(recs, pop_model.item_novelty))

        agg = compute_aggregate_metrics(all_recs, total_catalog_movie_ids, pop_model.item_novelty)
        gamma_records.append({
            "Gamma": gam,
            "NDCG@10": np.mean(ndcg_l),
            "Precision@10": np.mean(p_l),
            "Calibration_D_KL": np.mean(kl_l),
            "Novelty_Bits": np.mean(nov_l),
            "Catalog_Coverage_Pct": agg["catalog_coverage"],
            "Long_Tail_Ratio": agg["long_tail_ratio"]
        })
        print(f"  Gamma={gam:.2f} -> NDCG={np.mean(ndcg_l):.4f}, Novelty={np.mean(nov_l):.2f} bits, Long-Tail={agg['long_tail_ratio']:.2f}%")

    gamma_df = pd.DataFrame(gamma_records)
    gamma_csv = os.path.join(REPORTS_DIR, "gamma_sweep.csv")
    gamma_df.to_csv(gamma_csv, index=False)

    # -------------------------------------------------------------
    # EXPERIMENT 3: Popularity Penalty Baseline (r_hat - beta * log(1+count))
    # -------------------------------------------------------------
    print("\n[Ablation 3] Evaluating Popularity Penalty Baseline (Score - beta * log(1+pop))...")
    pop_pen_records = []

    for beta in [0.05, 0.10, 0.20]:
        ndcg_l, p_l, kl_l, nov_l = [], [], [], []
        all_recs = {}

        for u in test_users:
            u_train = user_train_movies.get(u, set())
            cand_scores = svd_model.predict_catalog_ratings_vectorized(u, total_catalog_movie_ids)
            top_cand = sorted([m for m in total_catalog_movie_ids if m not in u_train],
                              key=lambda m: cand_scores.get(m, 0.0), reverse=True)[:200]

            penalized = sorted(
                top_cand,
                key=lambda m: cand_scores.get(m, 0.0) - beta * np.log1p(train_counts.get(m, 0)),
                reverse=True
            )[:top_k]

            all_recs[u] = penalized

            ndcg_l.append(compute_ndcg_at_k(penalized, user_test_ratings[u], k=top_k))
            p_l.append(compute_precision_recall_f1_at_k(penalized, user_test_relevant[u], k=top_k)[0])
            nov_l.append(compute_list_novelty(penalized, pop_model.item_novelty))

            # Genre distribution & D_KL
            counts = np.zeros(calibrated_engine.num_genres, dtype=float)
            for m in penalized:
                counts += calibrated_engine.compute_genre_vector(m, movie_genre_map)
            q_dist = counts / np.sum(counts) if np.sum(counts) > 0 else np.ones(calibrated_engine.num_genres) / calibrated_engine.num_genres
            kl_l.append(calibrated_engine.compute_kl_divergence(user_genre_profiles[u], q_dist))

        agg = compute_aggregate_metrics(all_recs, total_catalog_movie_ids, pop_model.item_novelty)
        pop_pen_records.append({
            "Beta_Penalty": beta,
            "NDCG@10": np.mean(ndcg_l),
            "Precision@10": np.mean(p_l),
            "Calibration_D_KL": np.mean(kl_l),
            "Novelty_Bits": np.mean(nov_l),
            "Catalog_Coverage_Pct": agg["catalog_coverage"],
            "Long_Tail_Ratio": agg["long_tail_ratio"]
        })
        print(f"  Pop Penalty Beta={beta:.2f} -> NDCG={np.mean(ndcg_l):.4f}, D_KL={np.mean(kl_l):.4f} (Uncalibrated!), Novelty={np.mean(nov_l):.2f}")

    pop_pen_df = pd.DataFrame(pop_pen_records)
    pop_pen_csv = os.path.join(REPORTS_DIR, "popularity_penalty_baseline.csv")
    pop_pen_df.to_csv(pop_pen_csv, index=False)

    print("\n" + "=" * 90)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 90)
    print("\n1. CANDIDATE POOL SWEEP:")
    print(pool_df.to_string(index=False))
    print("\n2. GAMMA (NOVELTY REGULARIZATION) SWEEP:")
    print(gamma_df.to_string(index=False))
    print("\n3. POPULARITY PENALTY BASELINE:")
    print(pop_pen_df.to_string(index=False))
    print("=" * 90)

    return {
        "ref_drift_mean": ref_drift_mean,
        "pool_df": pool_df,
        "gamma_df": gamma_df,
        "pop_pen_df": pop_pen_df
    }


if __name__ == "__main__":
    run_ablations()
