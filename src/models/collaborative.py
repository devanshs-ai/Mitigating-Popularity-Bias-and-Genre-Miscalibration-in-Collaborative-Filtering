"""
Collaborative Filtering Recommender using Latent Factor SVD (Singular Value Decomposition).
Implements Simon Funk's regularized matrix factorization:
r̂_ui = μ + b_u + b_i + q_i^T * p_u
"""

import numpy as np
from surprise import SVD, accuracy


class SVDRecommender:
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=random_state
        )
        self.trainset = None

    def fit(self, trainset):
        self.trainset = trainset
        print(f"[SVD] Training model with {self.n_factors} latent factors, {self.n_epochs} epochs...")
        self.model.fit(trainset)
        print("[SVD] Training complete.")
        return self

    def predict_rating(self, user_id, movie_id):
        return self.model.predict(user_id, movie_id).est

    def evaluate_testset(self, testset):
        """Computes RMSE and MAE against testset tuples (userId, movieId, rating)."""
        predictions = self.model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        return {"rmse": rmse, "mae": mae, "predictions": predictions}

    def predict_catalog_ratings_vectorized(self, user_id, catalog_movie_ids):
        """Ultra-fast vectorized rating prediction across catalog items."""
        try:
            u_inner = self.trainset.to_inner_uid(user_id)
            mu = self.trainset.global_mean
            bu = self.model.bu[u_inner]
            pu = self.model.pu[u_inner]
            inner_preds = np.clip(mu + bu + self.model.bi + np.dot(self.model.qi, pu), 0.5, 5.0)

            scores = {}
            for i_inner in range(self.trainset.n_items):
                raw_id = self.trainset.to_raw_iid(i_inner)
                scores[raw_id] = float(inner_preds[i_inner])

            # Fallback for catalog movies with no train ratings
            for m_id in catalog_movie_ids:
                if m_id not in scores:
                    scores[m_id] = mu
            return scores
        except (ValueError, KeyError):
            mu = self.trainset.global_mean if self.trainset else 3.5
            return {m_id: mu for m_id in catalog_movie_ids}

    def recommend(self, user_id, catalog_movie_ids, top_k=10, exclude_movie_ids=None):
        if exclude_movie_ids is None:
            exclude_movie_ids = set()

        scores_dict = self.predict_catalog_ratings_vectorized(user_id, catalog_movie_ids)
        candidates = [m_id for m_id in catalog_movie_ids if m_id not in exclude_movie_ids]

        ranked = sorted(candidates, key=lambda m: scores_dict.get(m, 0.0), reverse=True)
        return ranked[:top_k]
