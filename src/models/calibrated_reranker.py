"""
Calibrated & Long-Tail Aware Re-ranking Engine.
Pops the filter bubble by aligning recommendation genre distributions
with the user's authentic historical taste using Kullback-Leibler divergence,
while promoting long-tail items through self-information novelty weighting.

Formulation:
i* = argmax_{i in C \ R} [ (1 - lambda) * relevance(i) 
                          - lambda * D_KL( P(u) || Q(R U {i}) ) 
                          + gamma * Novelty(i) ]
"""

import numpy as np


class CalibratedReranker:
    def __init__(self, all_genres, lambda_param=0.5, gamma_param=0.05, epsilon=1e-5):
        self.all_genres = all_genres
        self.num_genres = len(all_genres)
        self.genre_to_idx = {g: i for i, g in enumerate(all_genres)}
        self.lambda_param = lambda_param
        self.gamma_param = gamma_param
        self.epsilon = epsilon

    def compute_genre_vector(self, movie_id, movie_genre_map):
        """Returns binary indicator vector of genres present in movie."""
        vec = np.zeros(self.num_genres, dtype=float)
        genres = movie_genre_map.get(movie_id, [])
        for g in genres:
            if g in self.genre_to_idx:
                vec[self.genre_to_idx[g]] = 1.0
        return vec

    def compute_kl_divergence(self, p_dist, q_dist):
        """Computes D_KL(P || Q) with Laplace smoothing."""
        p_smooth = (p_dist + self.epsilon)
        p_smooth /= np.sum(p_smooth)

        q_smooth = (q_dist + self.epsilon)
        q_smooth /= np.sum(q_smooth)

        return np.sum(p_smooth * np.log(p_smooth / q_smooth))

    def rerank(self, candidate_items, candidate_scores, user_genre_profile,
               movie_genre_map, item_novelty=None, top_k=10, lambda_param=None, gamma_param=None):
        """
        Greedy calibrated re-ranking algorithm.
        candidate_items: list of movie IDs
        candidate_scores: dict of {movieId: score} (e.g. SVD predicted ratings)
        user_genre_profile: P(g|u) normalized vector of length num_genres
        movie_genre_map: dict of {movieId: list_of_genres}
        item_novelty: dict of {movieId: self_information_bits}
        """
        if lambda_param is None:
            lambda_param = self.lambda_param
        if gamma_param is None:
            gamma_param = self.gamma_param

        # Precompute movie genre vectors
        movie_genre_vecs = {
            m_id: self.compute_genre_vector(m_id, movie_genre_map)
            for m_id in candidate_items
        }

        # Min-max normalize candidate scores to [0, 1]
        scores_arr = np.array([candidate_scores[m] for m in candidate_items])
        s_min, s_max = scores_arr.min(), scores_arr.max()
        if s_max > s_min:
            norm_scores = {m: (candidate_scores[m] - s_min) / (s_max - s_min) for m in candidate_items}
        else:
            norm_scores = {m: 0.5 for m in candidate_items}

        # Min-max normalize novelty scores
        if item_novelty is not None:
            nov_arr = np.array([item_novelty.get(m, 0.0) for m in candidate_items])
            n_min, n_max = nov_arr.min(), nov_arr.max()
            if n_max > n_min:
                norm_nov = {m: (item_novelty.get(m, 0.0) - n_min) / (n_max - n_min) for m in candidate_items}
            else:
                norm_nov = {m: 0.0 for m in candidate_items}
        else:
            norm_nov = {m: 0.0 for m in candidate_items}

        selected_items = []
        cumulative_genre_counts = np.zeros(self.num_genres, dtype=float)
        remaining_candidates = set(candidate_items)

        for step in range(min(top_k, len(candidate_items))):
            best_item = None
            best_obj_val = -1e9
            best_new_counts = None

            for m_id in remaining_candidates:
                # Add item's genres to hypothetical list
                hypo_counts = cumulative_genre_counts + movie_genre_vecs[m_id]
                total_genre_occurrences = np.sum(hypo_counts)

                if total_genre_occurrences > 0:
                    q_dist = hypo_counts / total_genre_occurrences
                else:
                    q_dist = np.ones(self.num_genres) / self.num_genres

                # Compute KL divergence from user true preference distribution
                kl_div = self.compute_kl_divergence(user_genre_profile, q_dist)

                # Objective: balance accuracy with calibration and novelty
                obj_val = (
                    (1.0 - lambda_param) * norm_scores[m_id]
                    - lambda_param * kl_div
                    + gamma_param * norm_nov[m_id]
                )

                if obj_val > best_obj_val:
                    best_obj_val = obj_val
                    best_item = m_id
                    best_new_counts = hypo_counts

            selected_items.append(best_item)
            cumulative_genre_counts = best_new_counts
            remaining_candidates.remove(best_item)

        # Compute final list distribution and calibration error
        total_occurrences = np.sum(cumulative_genre_counts)
        if total_occurrences > 0:
            final_q = cumulative_genre_counts / total_occurrences
        else:
            final_q = np.ones(self.num_genres) / self.num_genres

        final_kl = self.compute_kl_divergence(user_genre_profile, final_q)

        return {
            "recommended_items": selected_items,
            "final_genre_dist": final_q,
            "calibration_error": final_kl
        }
