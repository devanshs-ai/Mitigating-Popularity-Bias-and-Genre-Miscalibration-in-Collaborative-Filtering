"""
Comprehensive Recommender Evaluation Suite.
Implements:
1. Ranking metrics: Precision@K, Recall@K, F1@K, NDCG@K, MAP@K
2. Diversity & Novelty: Intra-List Diversity (ILD), Self-Information Novelty (bits)
3. Catalog & Exposure: Aggregate Catalog Coverage (%), Gini Coefficient of Item Exposure
4. Calibration: Kullback-Leibler Divergence D_KL(P || Q)
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_precision_recall_f1_at_k(recommended_items, relevant_items, k=10):
    """
    recommended_items: ordered list of recommended movie IDs
    relevant_items: set of ground-truth relevant movie IDs (e.g. rating >= 4.0 in testset)
    """
    rec_k = recommended_items[:k]
    hits = len(set(rec_k).intersection(relevant_items))

    precision = hits / float(k)
    recall = hits / float(len(relevant_items)) if len(relevant_items) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def compute_ndcg_at_k(recommended_items, relevant_items_dict, k=10):
    """
    relevant_items_dict: dict of {movieId: true_rating} for items rated by user in test set
    Relevance gain: 2^r - 1 (or 1 if r >= 4.0 else 0)
    """
    rec_k = recommended_items[:k]
    dcg = 0.0

    for i, m_id in enumerate(rec_k):
        rel = 1.0 if relevant_items_dict.get(m_id, 0.0) >= 4.0 else 0.0
        if rel > 0:
            dcg += (np.power(2.0, rel) - 1.0) / np.log2(i + 2.0)

    # Ideal DCG
    ideal_rels = sorted([1.0 if r >= 4.0 else 0.0 for r in relevant_items_dict.values()], reverse=True)[:k]
    idcg = sum((np.power(2.0, r) - 1.0) / np.log2(i + 2.0) for i, r in enumerate(ideal_rels))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_map_at_k(recommended_items, relevant_items, k=10):
    """Mean Average Precision at K."""
    if not relevant_items:
        return 0.0

    rec_k = recommended_items[:k]
    score = 0.0
    num_hits = 0.0

    for i, m_id in enumerate(rec_k):
        if m_id in relevant_items:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(relevant_items), k)


def compute_intra_list_diversity(recommended_items, tfidf_matrix, movie_id_to_idx):
    """
    Computes Intra-List Diversity (ILD) as 1 - mean pairwise cosine similarity.
    Higher value (closer to 1.0) indicates greater diversity among recommended items.
    """
    valid_indices = [movie_id_to_idx[m] for m in recommended_items if m in movie_id_to_idx]
    k = len(valid_indices)
    if k <= 1:
        return 0.0

    vectors = tfidf_matrix[valid_indices]
    sim_matrix = cosine_similarity(vectors)

    # Exclude self-similarity diagonal
    mask = np.triu(np.ones((k, k), dtype=bool), k=1)
    pairwise_sims = sim_matrix[mask]

    mean_sim = np.mean(pairwise_sims) if len(pairwise_sims) > 0 else 0.0
    return 1.0 - mean_sim


def compute_list_novelty(recommended_items, item_novelty_dict):
    """Computes mean novelty (self-information in bits) of the recommended list."""
    novelties = [item_novelty_dict.get(m, 13.0) for m in recommended_items]
    return float(np.mean(novelties)) if novelties else 0.0


def compute_gini(array):
    """Computes the Gini inequality coefficient."""
    array = np.sort(np.asarray(array, dtype=float))
    n = len(array)
    if n == 0 or np.sum(array) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def compute_aggregate_metrics(all_recommendations_dict, total_catalog_movie_ids, item_novelty_dict):
    """
    Computes global exposure and catalog-level metrics across all users:
    - Aggregate Catalog Coverage %
    - Recommendation Gini Coefficient
    - Long-tail item recommendation ratio %
    """
    all_rec_items = []
    item_rec_counts = {m_id: 0 for m_id in total_catalog_movie_ids}

    for user_id, recs in all_recommendations_dict.items():
        all_rec_items.extend(recs)
        for m_id in recs:
            if m_id in item_rec_counts:
                item_rec_counts[m_id] += 1

    unique_recommended = set(all_rec_items)
    catalog_coverage = (len(unique_recommended) / len(total_catalog_movie_ids)) * 100.0

    counts_arr = np.array(list(item_rec_counts.values()))
    gini_coeff = compute_gini(counts_arr)

    # Long-tail threshold: items with novelty >= median novelty or top 80% least rated
    all_novs = np.array(list(item_novelty_dict.values()))
    tail_threshold = np.percentile(all_novs, 50)
    tail_recs = sum(1 for m_id in all_rec_items if item_novelty_dict.get(m_id, 0.0) >= tail_threshold)
    tail_ratio = (tail_recs / len(all_rec_items)) * 100.0 if all_rec_items else 0.0

    return {
        "catalog_coverage": catalog_coverage,
        "recommendation_gini": gini_coeff,
        "long_tail_ratio": tail_ratio,
        "total_unique_items_recommended": len(unique_recommended)
    }


def compute_total_variation_distance(p_dist, q_dist):
    """
    Computes Total Variation Distance (TVD / L1 distance / 2) between two distributions.
    Independent calibration metric that bounds the maximum difference in probability:
    TVD(P, Q) = 0.5 * sum(|p_g - q_g|)
    Range: [0.0 (identical), 1.0 (completely disjoint)]
    """
    p = np.asarray(p_dist, dtype=float)
    q = np.asarray(q_dist, dtype=float)
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q
    return float(0.5 * np.sum(np.abs(p - q)))


def compute_genre_entropy(q_dist):
    """Computes Shannon Entropy of recommended genre distribution: H(Q) = -sum(q * log2(q))."""
    q = np.asarray(q_dist, dtype=float)
    q = q / np.sum(q) if np.sum(q) > 0 else q
    pos = q[q > 0]
    return float(-np.sum(pos * np.log2(pos))) if len(pos) > 0 else 0.0


def compute_paired_ttest(metric_a, metric_b):
    """Computes two-sided paired Student's t-test between two arrays of user metrics."""
    from scipy.stats import ttest_rel
    stat, p_val = ttest_rel(metric_a, metric_b)
    return {"t_stat": float(stat), "p_value": float(p_val)}
