"""
Data loader and preprocessor for MovieLens Latest-Small dataset.
Automates dataset download, preprocessing, feature extraction, 
user genre profile generation, and stratified train/test split.
"""

import os
import re
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from collections import defaultdict
from surprise import Dataset, Reader

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def download_and_extract_movielens(raw_dir=RAW_DIR):
    """Downloads and extracts MovieLens Latest-Small if not already present."""
    os.makedirs(raw_dir, exist_ok=True)
    extract_path = os.path.join(raw_dir, "ml-latest-small")
    zip_path = os.path.join(raw_dir, "ml-latest-small.zip")

    if not os.path.exists(extract_path):
        print(f"[Dataset] Downloading MovieLens from {MOVIELENS_URL}...")
        import requests
        try:
            r = requests.get(MOVIELENS_URL, verify=True, stream=True)
        except Exception:
            r = requests.get(MOVIELENS_URL, verify=False, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("[Dataset] Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        print("[Dataset] Extraction complete.")
    else:
        print("[Dataset] Found existing MovieLens dataset in raw directory.")

    return extract_path


def load_raw_data(data_dir=None):
    """Loads raw ratings, movies, and tags dataframes."""
    if data_dir is None:
        data_dir = download_and_extract_movielens()
    
    ratings_df = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    movies_df = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    tags_df = pd.read_csv(os.path.join(data_dir, "tags.csv"))
    return ratings_df, movies_df, tags_df


def extract_year_and_clean_title(title):
    """Extracts 4-digit release year from title and returns (clean_title, year)."""
    match = re.search(r'\((\d{4})\)', title)
    if match:
        year = int(match.group(1))
        clean_title = re.sub(r'\s*\(\d{4}\)', '', title).strip()
    else:
        year = np.nan
        clean_title = title.strip()
    return clean_title, year


def preprocess_metadata(movies_df, tags_df):
    """
    Enriches movies with cleaned titles, release years, genre lists,
    and concatenated user tag strings for dense/TF-IDF text features.
    """
    cleaned_titles = []
    years = []
    for title in movies_df["title"]:
        c_title, yr = extract_year_and_clean_title(title)
        cleaned_titles.append(c_title)
        years.append(yr)

    movies_df = movies_df.copy()
    movies_df["clean_title"] = cleaned_titles
    movies_df["year"] = years

    # Parse genres into list
    movies_df["genre_list"] = movies_df["genres"].apply(
        lambda g: [x for x in g.split("|") if x != "(no genres listed)"]
    )

    # Aggregate user tags per movie
    tags_agg = tags_df.groupby("movieId")["tag"].apply(
        lambda tags: " ".join([str(t).lower() for t in tags if pd.notnull(t)])
    ).reset_index()
    tags_agg.rename(columns={"tag": "tag_corpus"}, inplace=True)

    movies_df = movies_df.merge(tags_agg, on="movieId", how="left")
    movies_df["tag_corpus"] = movies_df["tag_corpus"].fillna("")

    # Create unified text representation: clean_title + genres + tags
    movies_df["content_text"] = movies_df.apply(
        lambda row: f"{row['clean_title'].lower()} {' '.join(row['genre_list']).lower()} {row['tag_corpus']}",
        axis=1
    )

    return movies_df


def get_all_genres(movies_df):
    """Returns sorted list of all unique genres."""
    all_genres = set()
    for g_list in movies_df["genre_list"]:
        all_genres.update(g_list)
    return sorted(list(all_genres))


def compute_user_genre_profiles(train_ratings_df, movies_df, all_genres=None):
    """
    Computes normalized genre preference distribution P(g|u) for each user
    based on their historical ratings (weighted by rating score).
    """
    if all_genres is None:
        all_genres = get_all_genres(movies_df)

    movie_genre_map = movies_df.set_index("movieId")["genre_list"].to_dict()
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}
    num_genres = len(all_genres)

    user_profiles = {}
    grouped = train_ratings_df.groupby("userId")

    for user_id, group in grouped:
        genre_weights = np.zeros(num_genres, dtype=float)
        total_weight = 0.0

        for _, row in group.iterrows():
            m_id = row["movieId"]
            rating = row["rating"]
            genres = movie_genre_map.get(m_id, [])
            for g in genres:
                if g in genre_to_idx:
                    # Weight by positive rating (e.g. rating/5.0)
                    genre_weights[genre_to_idx[g]] += (rating / 5.0)
                    total_weight += (rating / 5.0)

        # Normalize with Laplace smoothing
        if total_weight > 0:
            user_profiles[user_id] = (genre_weights + 1e-5) / (total_weight + 1e-5 * num_genres)
        else:
            # Uniform fallback for zero-history users
            user_profiles[user_id] = np.ones(num_genres) / num_genres

    return user_profiles, all_genres


def create_stratified_split(ratings_df, test_ratio=0.2, random_state=42):
    """
    Performs stratified 80/20 train/test split per user, ensuring:
    1. Every user is represented in both train and test sets.
    2. Strict temporal/pseudo-random reproducibility.
    """
    np.random.seed(random_state)
    train_indices = []
    test_indices = []

    for user_id, group in ratings_df.groupby("userId"):
        indices = group.index.values
        n_ratings = len(indices)

        if n_ratings <= 2:
            # For users with very few ratings, keep at least 1 in train
            train_indices.extend(indices[:-1])
            test_indices.append(indices[-1])
        else:
            n_test = max(1, int(round(n_ratings * test_ratio)))
            shuffled = np.random.permutation(indices)
            test_indices.extend(shuffled[:n_test])
            train_indices.extend(shuffled[n_test:])

    train_df = ratings_df.loc[train_indices].reset_index(drop=True)
    test_df = ratings_df.loc[test_indices].reset_index(drop=True)
    return train_df, test_df


def prepare_surprise_dataset(train_df, test_df):
    """Prepares scikit-surprise Dataset and trainset/testset structures."""
    reader = Reader(rating_scale=(0.5, 5.0))
    trainset = Dataset.load_from_df(train_df[["userId", "movieId", "rating"]], reader).build_full_trainset()
    testset = [tuple(x) for x in test_df[["userId", "movieId", "rating"]].to_numpy()]
    return trainset, testset


def load_dataset_pipeline(random_state=42):
    """End-to-end data pipeline returning all processed components."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    ratings_df, movies_df, tags_df = load_raw_data()
    movies_df = preprocess_metadata(movies_df, tags_df)
    train_df, test_df = create_stratified_split(ratings_df, test_ratio=0.2, random_state=random_state)
    all_genres = get_all_genres(movies_df)
    user_genre_profiles, _ = compute_user_genre_profiles(train_df, movies_df, all_genres)
    trainset, testset = prepare_surprise_dataset(train_df, test_df)

    return {
        "ratings_df": ratings_df,
        "movies_df": movies_df,
        "tags_df": tags_df,
        "train_df": train_df,
        "test_df": test_df,
        "all_genres": all_genres,
        "user_genre_profiles": user_genre_profiles,
        "trainset": trainset,
        "testset": testset
    }


if __name__ == "__main__":
    data = load_dataset_pipeline()
    print("Dataset Pipeline loaded successfully!")
    print(f"Users: {data['ratings_df']['userId'].nunique()}")
    print(f"Movies: {data['movies_df']['movieId'].nunique()}")
    print(f"Total Ratings: {len(data['ratings_df'])}")
    print(f"Train Ratings: {len(data['train_df'])}, Test Ratings: {len(data['test_df'])}")
    print(f"Unique Genres: {len(data['all_genres'])} -> {data['all_genres']}")
