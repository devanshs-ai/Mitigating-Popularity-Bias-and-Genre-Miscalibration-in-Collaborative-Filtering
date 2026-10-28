"""
Content-Based Recommender using TF-IDF features and Cosine Similarity.
Represents movies via text descriptions (clean title + genres + user tags).
Supports both seed-movie recommendations and user-profile centroid recommendations.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features
        )
        self.tfidf_matrix = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.movies_df = None

    def fit(self, movies_df):
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.movie_id_to_idx = {m_id: i for i, m_id in enumerate(self.movies_df["movieId"])}
        self.idx_to_movie_id = {i: m_id for i, m_id in enumerate(self.movies_df["movieId"])}

        # Fit TF-IDF on enriched content text
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df["content_text"])
        return self

    def get_seed_similarities(self, seed_movie_id):
        """Returns array of cosine similarities for all catalog movies to seed movie."""
        if seed_movie_id not in self.movie_id_to_idx:
            return np.zeros(len(self.movies_df))
        seed_idx = self.movie_id_to_idx[seed_movie_id]
        seed_vec = self.tfidf_matrix[seed_idx]
        sims = cosine_similarity(seed_vec, self.tfidf_matrix).flatten()
        return sims

    def get_user_profile_vector(self, user_ratings_df, min_rating=3.0):
        """Constructs user preference centroid by weighting movie TF-IDF vectors by user rating."""
        liked = user_ratings_df[user_ratings_df["rating"] >= min_rating]
        if liked.empty:
            liked = user_ratings_df

        valid_indices = []
        weights = []
        for _, row in liked.iterrows():
            m_id = row["movieId"]
            if m_id in self.movie_id_to_idx:
                valid_indices.append(self.movie_id_to_idx[m_id])
                weights.append(row["rating"] / 5.0)

        if not valid_indices:
            return None

        vectors = self.tfidf_matrix[valid_indices].toarray()
        weights = np.array(weights).reshape(-1, 1)
        profile_vec = np.sum(vectors * weights, axis=0, keepdims=True)
        norm = np.linalg.norm(profile_vec)
        if norm > 0:
            profile_vec /= norm
        return profile_vec

    def recommend(self, user_ratings_df=None, seed_movie_id=None, top_k=10, exclude_movie_ids=None):
        if exclude_movie_ids is None:
            exclude_movie_ids = set()

        if seed_movie_id is not None and seed_movie_id in self.movie_id_to_idx:
            sims = self.get_seed_similarities(seed_movie_id)
        elif user_ratings_df is not None and not user_ratings_df.empty:
            profile_vec = self.get_user_profile_vector(user_ratings_df)
            if profile_vec is not None:
                sims = cosine_similarity(profile_vec, self.tfidf_matrix).flatten()
            else:
                sims = np.zeros(len(self.movies_df))
        else:
            sims = np.zeros(len(self.movies_df))

        # Sort descending
        ranked_indices = np.argsort(-sims)
        recs = []
        for idx in ranked_indices:
            m_id = self.idx_to_movie_id[idx]
            if m_id not in exclude_movie_ids:
                recs.append(m_id)
                if len(recs) == top_k:
                    break
        return recs
