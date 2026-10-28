"""
Task 2: Exploratory Data Analysis & Long-Tail Bias Profiling.
Generates statistical summaries, long-tail item distributions, 
Gini coefficient of dataset exposure, and high-resolution figures.
"""

import os
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_dataset_pipeline

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Publication plot styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})


def compute_gini(array):
    """Computes the Gini coefficient of inequality for an array of values."""
    array = np.sort(np.asarray(array, dtype=float))
    n = len(array)
    if n == 0 or np.sum(array) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def run_eda():
    print("[EDA] Loading preprocessed pipeline data...")
    data = load_dataset_pipeline()
    ratings_df = data["ratings_df"]
    movies_df = data["movies_df"]

    n_users = ratings_df["userId"].nunique()
    n_movies = ratings_df["movieId"].nunique()
    n_ratings = len(ratings_df)
    sparsity = (1.0 - (n_ratings / (n_users * n_movies))) * 100.0

    print("=" * 60)
    print("DATASET PROFILE & SPARSITY SUMMARY")
    print(f"Users: {n_users}")
    print(f"Active Movies Rated: {n_movies} (Catalog size: {len(movies_df)})")
    print(f"Total Ratings: {n_ratings}")
    print(f"Matrix Sparsity: {sparsity:.2f}%")
    print(f"Mean Rating: {ratings_df['rating'].mean():.3f}")
    print(f"Median Rating: {ratings_df['rating'].median():.1f}")
    print(f"Rating Std Dev: {ratings_df['rating'].std():.3f}")

    user_counts = ratings_df.groupby("userId").size()
    movie_counts = ratings_df.groupby("movieId").size()

    print(f"Ratings per user: Mean={user_counts.mean():.1f}, Median={user_counts.median():.1f}, Max={user_counts.max()}")
    print(f"Ratings per movie: Mean={movie_counts.mean():.1f}, Median={movie_counts.median():.1f}, Max={movie_counts.max()}")

    # Item popularity inequality
    item_gini = compute_gini(movie_counts.values)
    sorted_counts = movie_counts.sort_values(ascending=False).values
    cumsum_ratings = np.cumsum(sorted_counts) / n_ratings
    top_20_pct_idx = int(round(0.20 * len(sorted_counts)))
    top_20_share = cumsum_ratings[top_20_pct_idx - 1] * 100.0

    print(f"Item Exposure Gini Coefficient: {item_gini:.4f}")
    print(f"Top 20% movies account for {top_20_share:.2f}% of all interaction volume!")
    print("=" * 60)

    # -------------------------------------------------------------
    # Figure 1: Rating Distribution + Long-Tail Item Popularity Curve
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Ax 1: Rating score histogram
    palette = sns.color_palette("mako", 10)
    rating_counts = ratings_df["rating"].value_counts().sort_index()
    bars = ax1.bar(rating_counts.index, rating_counts.values, width=0.35, color=palette[4], edgecolor="black", alpha=0.85)
    ax1.set_xlabel("Rating Score (Stars)")
    ax1.set_ylabel("Rating Frequency")
    ax1.set_title("(a) Rating Score Distribution")
    ax1.set_xticks(np.arange(0.5, 5.5, 0.5))
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate percentage above each bar
    for bar in bars:
        height = bar.get_height()
        pct = (height / n_ratings) * 100
        if pct > 2.0:
            ax1.annotate(f"{pct:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8)

    # Ax 2: Cumulative interaction vs Item catalog share (Lorenz Curve / Long Tail)
    x_items = np.linspace(0, 100, len(sorted_counts))
    ax2.plot(x_items, cumsum_ratings * 100.0, color="#1f77b4", linewidth=2.5, label="Observed Cumulative Ratings")
    ax2.plot([0, 100], [0, 100], color="gray", linestyle="--", label="Perfect Equality (No Bias)")
    ax2.axvline(20, color="#d62728", linestyle=":", label=f"Top 20% Items ({top_20_share:.1f}% Ratings)")
    ax2.scatter([20], [top_20_share], color="#d62728", s=50, zorder=5)

    ax2.set_xlabel("Percentage of Catalog Items (Ranked by Popularity)")
    ax2.set_ylabel("Cumulative % of Total Ratings")
    ax2.set_title(f"(b) Long-Tail Popularity Skew (Gini = {item_gini:.3f})")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "fig1_eda_rating_and_longtail.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"[EDA] Saved Figure 1 to: {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Genre Volume and Mean Rating Analysis
    # -------------------------------------------------------------
    genre_data = []
    # Merge ratings with movie genres
    ratings_with_movies = ratings_df.merge(movies_df[["movieId", "genre_list"]], on="movieId")

    genre_counts = defaultdict(int)
    genre_ratings = defaultdict(list)

    for _, row in ratings_with_movies.iterrows():
        r = row["rating"]
        for g in row["genre_list"]:
            genre_counts[g] += 1
            genre_ratings[g].append(r)

    genre_summary = pd.DataFrame([
        {
            "Genre": g,
            "Count": genre_counts[g],
            "Mean_Rating": np.mean(genre_ratings[g])
        }
        for g in genre_counts
    ]).sort_values("Count", ascending=False)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    x = np.arange(len(genre_summary))
    width = 0.55

    bars = ax1.bar(x, genre_summary["Count"], width=width, color="#4c72b0", alpha=0.8, edgecolor="black", label="Number of Ratings")
    line = ax2.plot(x, genre_summary["Mean_Rating"], color="#c44e52", marker="o", linewidth=2.2, label="Mean Rating")

    ax1.set_xlabel("Movie Genre")
    ax1.set_ylabel("Total Interaction Count", color="#4c72b0")
    ax2.set_ylabel("Mean Rating (Stars)", color="#c44e52")
    ax1.set_xticks(x)
    ax1.set_xticklabels(genre_summary["Genre"], rotation=45, ha="right")
    ax2.set_ylim(2.5, 4.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.set_title("Genre Interaction Volume and Average User Rating")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "fig2_eda_genre_distribution.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"[EDA] Saved Figure 2 to: {fig2_path}")

    return {
        "n_users": n_users,
        "n_movies": n_movies,
        "n_ratings": n_ratings,
        "sparsity": sparsity,
        "item_gini": item_gini,
        "top_20_share": top_20_share
    }


if __name__ == "__main__":
    run_eda()
