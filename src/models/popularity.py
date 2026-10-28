"""
Bayesian Weighted Rating Popularity Recommender.
Implements the IMDb formula:
WR = (v / (v + m)) * R + (m / (v + m)) * C
where:
- v = vote count for movie
- m = minimum votes required (90th percentile)
- R = average rating of movie
- C = global mean rating across dataset
"""

import numpy as np
import pandas as pd


class PopularityRecommender:
    def __init__(self, percentile=0.90):
        self.percentile = percentile
        self.global_mean_C = 0.0
        self.min_ratings_m = 0.0
        self.ranked_catalog = None
        self.movie_scores = {}
        self.item_probabilities = {}
        self.item_novelty = {}

    def fit(self, train_df, movies_df):
        total_ratings = len(train_df)
        self.global_mean_C = train_df["rating"].mean()

        # Group by movie
        movie_stats = train_df.groupby("movieId")["rating"].agg(["count", "mean"]).reset_index()
        movie_stats.columns = ["movieId", "vote_count", "mean_rating"]

        # 90th percentile threshold
        self.min_ratings_m = movie_stats["vote_count"].quantile(self.percentile)

        # Bayesian weighted rating
        def calc_wr(row):
            v = row["vote_count"]
            R = row["mean_rating"]
            m = self.min_ratings_m
            C = self.global_mean_C
            return (v / (v + m)) * R + (m / (v + m)) * C

        movie_stats["weighted_rating"] = movie_stats.apply(calc_wr, axis=1)

        # Merge with full catalog (unrated movies get C * (0/(0+m)) + C = default C)
        catalog = movies_df[["movieId", "clean_title", "genres", "genre_list"]].merge(
            movie_stats, on="movieId", how="left"
        )
        catalog["vote_count"] = catalog["vote_count"].fillna(0)
        catalog["mean_rating"] = catalog["mean_rating"].fillna(self.global_mean_C)
        catalog["weighted_rating"] = catalog["weighted_rating"].fillna(
            (0 / (0 + self.min_ratings_m)) * self.global_mean_C + (self.min_ratings_m / (0 + self.min_ratings_m)) * self.global_mean_C
        )

        catalog.sort_values("weighted_rating", ascending=False, inplace=True)
        catalog.reset_index(drop=True, inplace=True)

        self.ranked_catalog = catalog
        self.movie_scores = catalog.set_index("movieId")["weighted_rating"].to_dict()

        # Compute item novelty (Self-Information: -log2(p(i)))
        for _, row in catalog.iterrows():
            m_id = row["movieId"]
            cnt = row["vote_count"]
            p_i = max(cnt, 1) / total_ratings
            self.item_probabilities[m_id] = p_i
            self.item_novelty[m_id] = -np.log2(p_i)

        return self

    def recommend(self, user_id=None, top_k=10, exclude_movie_ids=None):
        if exclude_movie_ids is None:
            exclude_movie_ids = set()

        recs = []
        for _, row in self.ranked_catalog.iterrows():
            m_id = row["movieId"]
            if m_id not in exclude_movie_ids:
                recs.append(m_id)
                if len(recs) == top_k:
                    break
        return recs
