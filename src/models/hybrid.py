"""
Hybrid Recommender & Cold-Start Router.
Combines min-max normalized content similarity with SVD predicted rating:
score(i) = alpha * ĉ(i) + (1 - alpha) * ((r̂_ui - 0.5) / 4.5)
Implements intelligent cold-start routing for unseen users and items.
"""

import numpy as np


class HybridRecommender:
    def __init__(self, popularity_model, content_model, svd_model, alpha=0.6):
        self.popularity_model = popularity_model
        self.content_model = content_model
        self.svd_model = svd_model
        self.alpha = alpha

    def get_candidate_scores(self, user_id, user_ratings_df, seed_movie_id, candidate_movie_ids, alpha=None):
        if alpha is None:
            alpha = self.alpha

        # Compute content similarities
        if seed_movie_id is not None:
            content_sims_all = self.content_model.get_seed_similarities(seed_movie_id)
            c_scores = {
                m_id: content_sims_all[self.content_model.movie_id_to_idx[m_id]]
                if m_id in self.content_model.movie_id_to_idx else 0.0
                for m_id in candidate_movie_ids
            }
        elif user_ratings_df is not None and not user_ratings_df.empty:
            profile_vec = self.content_model.get_user_profile_vector(user_ratings_df)
            if profile_vec is not None:
                sims = np.dot(self.content_model.tfidf_matrix.toarray(), profile_vec.T).flatten()
                c_scores = {
                    m_id: sims[self.content_model.movie_id_to_idx[m_id]]
                    if m_id in self.content_model.movie_id_to_idx else 0.0
                    for m_id in candidate_movie_ids
                }
            else:
                c_scores = {m_id: 0.0 for m_id in candidate_movie_ids}
        else:
            c_scores = {m_id: 0.0 for m_id in candidate_movie_ids}

        # Min-max normalize content similarities
        c_vals = np.array(list(c_scores.values()))
        c_min, c_max = c_vals.min(), c_vals.max()
        if c_max > c_min:
            norm_c = {m_id: (c_scores[m_id] - c_min) / (c_max - c_min) for m_id in candidate_movie_ids}
        else:
            norm_c = {m_id: 0.5 for m_id in candidate_movie_ids}

        # SVD predicted rating normalized from [0.5, 5.0] to [0.0, 1.0]
        svd_preds_dict = self.svd_model.predict_catalog_ratings_vectorized(user_id, candidate_movie_ids)
        hybrid_scores = {}
        for m_id in candidate_movie_ids:
            pred_r = svd_preds_dict.get(m_id, 3.5)
            norm_svd = max(0.0, min(1.0, (pred_r - 0.5) / 4.5))
            score = alpha * norm_c[m_id] + (1.0 - alpha) * norm_svd
            hybrid_scores[m_id] = {
                "hybrid_score": score,
                "norm_content": norm_c[m_id],
                "norm_svd": norm_svd,
                "predicted_rating": pred_r
            }

        return hybrid_scores

    def recommend(self, user_id=None, user_ratings_df=None, seed_movie_id=None,
                  catalog_movie_ids=None, top_k=10, exclude_movie_ids=None, alpha=None):
        if exclude_movie_ids is None:
            exclude_movie_ids = set()

        candidates = [m_id for m_id in catalog_movie_ids if m_id not in exclude_movie_ids]
        is_existing_user = (user_ratings_df is not None and not user_ratings_df.empty)

        # Cold-Start Router
        if not is_existing_user:
            if seed_movie_id is not None:
                # Cold user with seed -> Content-based
                return self.content_model.recommend(
                    seed_movie_id=seed_movie_id, top_k=top_k, exclude_movie_ids=exclude_movie_ids
                )
            else:
                # Cold user without seed -> Popularity
                return self.popularity_model.recommend(
                    top_k=top_k, exclude_movie_ids=exclude_movie_ids
                )
        else:
            # Existing user -> Hybrid blend
            scores = self.get_candidate_scores(
                user_id=user_id,
                user_ratings_df=user_ratings_df,
                seed_movie_id=seed_movie_id,
                candidate_movie_ids=candidates,
                alpha=alpha
            )
            ranked = sorted(scores.items(), key=lambda kv: kv[1]["hybrid_score"], reverse=True)
            return [m_id for m_id, _ in ranked[:top_k]]
