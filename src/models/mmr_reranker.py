"""
Maximal Marginal Relevance (MMR) Re-ranker (Carbonell & Goldstein, 1998).
Classic diversity baseline balancing relevance with novelty/dissimilarity to previously selected items:
i* = argmax_{i in C \ R} [ theta * score(i) - (1 - theta) * max_{j in R} Sim(i, j) ]
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MMRReranker:
    def __init__(self, theta=0.6):
        self.theta = theta

    def rerank(self, candidate_items, candidate_scores, tfidf_matrix, movie_id_to_idx, top_k=10, theta=None):
        if theta is None:
            theta = self.theta

        if not candidate_items:
            return []

        # Min-max normalize candidate scores to [0, 1]
        scores_arr = np.array([candidate_scores[m] for m in candidate_items])
        s_min, s_max = scores_arr.min(), scores_arr.max()
        if s_max > s_min:
            norm_scores = {m: (candidate_scores[m] - s_min) / (s_max - s_min) for m in candidate_items}
        else:
            norm_scores = {m: 0.5 for m in candidate_items}

        # Precompute dense cosine similarity matrix for candidate pool (100x100)
        cand_idx_map = {m: i for i, m in enumerate(candidate_items)}
        valid_indices = [movie_id_to_idx.get(m, 0) for m in candidate_items]
        cand_sub_matrix = tfidf_matrix[valid_indices]
        sim_matrix = cosine_similarity(cand_sub_matrix, cand_sub_matrix)

        selected = []
        remaining = list(range(len(candidate_items)))

        # First item is highest score
        first_idx = 0
        selected.append(first_idx)
        remaining.remove(first_idx)

        while len(selected) < min(top_k, len(candidate_items)):
            best_idx = None
            best_mmr_score = -1e9

            for c_idx in remaining:
                cand_score = norm_scores[candidate_items[c_idx]]
                max_sim = float(np.max(sim_matrix[c_idx, selected])) if selected else 0.0
                mmr_val = theta * cand_score - (1.0 - theta) * max_sim

                if mmr_val > best_mmr_score:
                    best_mmr_score = mmr_val
                    best_idx = c_idx

            if best_idx is None:
                best_idx = remaining[0]

            selected.append(best_idx)
            remaining.remove(best_idx)

        return [candidate_items[i] for i in selected]
