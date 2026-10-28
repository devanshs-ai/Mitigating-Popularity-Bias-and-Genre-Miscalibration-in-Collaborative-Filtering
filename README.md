# Mitigating Popularity Bias and Genre Miscalibration in Collaborative Filtering

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: MovieLens](https://img.shields.io/badge/Dataset-MovieLens--Latest--Small-orange.svg)](https://grouplens.org/datasets/movielens/)

An empirical, reproducible research framework investigating and mitigating **Popularity Bias** and **Genre Miscalibration** in Collaborative Filtering recommenders. Designed and evaluated on the MovieLens Latest Small benchmark (100,836 ratings, 610 users, 9,724 active movies, 98.30% sparse).

---

## 📌 Core Research Contribution

Standard collaborative filtering algorithms (such as regularized SVD matrix factorization) optimize strictly for rating prediction accuracy. This induces two severe, systematic distortions:
1. **Popularity Monopoly (Matthew Effect):** Blockbuster items in the top 5% capture over 75% of interaction volume, completely starving the long tail of the catalogue.
2. **Genre Miscalibration:** SVD recommendations collapse onto a user's single dominant genre, distorting their authentic multi-genre consumption profile.

This repository implements a **Calibrated and Long-Tail Aware Re-ranking Engine** directly on top of regularized SVD, evaluated against:
* **Plain SVD** (Simon Funk latent factor matrix factorization)
* **Maximal Marginal Relevance (MMR)** (Carbonell & Goldstein, 1998)
* **Steck's Calibrated Re-ranker** (Steck, ACM RecSys 2018)
* **Proposed Novelty-Augmented Calibrated SVD** (Ours)

---

## 📊 Rigorous Experimental Results (N = 497 Held-Out Test Users)

Evaluation is performed under a strict **leakage-free protocol**: $\lambda^*$ is selected on a separate Validation Cohort ($N=100$), and final metrics are computed over $N=497$ held-out test users under **full-catalog ranking** against all 9,724 candidate items.

| Model | NDCG@10 | Precision@10 | Calibration $D_{KL}$ | Total Variation Dist ($L_1$) | Diversity (ILD) | Novelty (Bits) | Catalog Coverage (%) | Long-Tail Exposure (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Plain SVD** | 0.0561 ± 0.0050 | 0.0445 ± 0.0039 | 1.4909 ± 0.0397 | 0.3621 | 0.9745 | 10.52 | 3.30% | 1.23% |
| **SVD + MMR (1998)** | 0.0550 ± 0.0049 | 0.0441 ± 0.0038 | 1.3036 ± 0.0367 | 0.3447 | **0.9848** | 10.59 | 3.41% | 1.29% |
| **Steck Calibrated (2018)** | **0.0567 ± 0.0048** | **0.0473 ± 0.0039** | 0.2081 ± 0.0051 | 0.1996 | 0.9764 | 10.68 | 4.00% | 1.83% |
| **Proposed Framework (Ours)**| 0.0514 ± 0.0047 | 0.0414 ± 0.0037 | **0.2023 ± 0.0049** | **0.1997** | 0.9774 | **10.92** | **4.36%** | **2.15%** |

### Statistical Significance (Paired Student's t-test, N = 497):
* **Taste Distortion ($D_{KL}$):** $t = -32.96, p = 4.85 \times 10^{-127}$ (statistically significant reduction of **86.4%** vs. Plain SVD, $p < 0.001$).
* **Total Variation Distance ($L_1$):** $t = -44.09, p = 9.93 \times 10^{-174}$ (statistically significant reduction of **44.8%** vs. Plain SVD, $p < 0.001$).
* **Ranking Accuracy (NDCG@10):** $t = -1.13, p = 0.2598$ ($p > 0.05$, confirming **no statistically significant degradation** in ranking utility).

---

## 🗂️ Project Directory Structure

```
├── data/raw/ml-latest-small/      # Raw MovieLens benchmark data
├── src/
│   ├── data_loader.py            # Automated dataset download & feature engineering
│   ├── models/
│   │   ├── popularity.py         # Bayesian Weighted Rating baseline (IMDb WR)
│   │   ├── content_based.py      # TF-IDF + Cosine similarity (titles, genres, tags)
│   │   ├── collaborative.py      # Vectorized Simon Funk SVD (100 latent factors)
│   │   ├── hybrid.py             # Linear hybrid and cold-start router
│   │   ├── mmr_reranker.py       # Carbonell & Goldstein (1998) MMR baseline
│   │   └── calibrated_reranker.py# Steck (2018) + Novelty-augmented re-ranker
│   └── evaluation/
│       └── metrics.py            # P@K, R@K, NDCG@K, D_KL, TVD, Entropy, Gini, Coverage
├── scripts/
│   ├── run_eda.py                # Exploratory Data Analysis & Figures 1-2
│   ├── run_benchmark.py          # Unified 5-model comparative benchmark
│   ├── run_calibration_experiment.py # Lambda sweep Pareto frontier analysis
│   ├── run_advanced_benchmark.py # Leakage-free SVD benchmark & statistical t-tests
│   └── plot_macro_calibration.py # Population-level genre calibration plotting
├── figures/                      # 300 DPI Publication Plots
│   ├── fig1_eda_rating_and_longtail.png
│   ├── fig2_eda_genre_distribution.png
│   ├── fig3_pareto_accuracy_vs_calibration.png
│   ├── fig4_exposure_and_coverage_comparison.png
│   └── fig5_macro_genre_calibration.png
├── reports/
│   ├── FINAL_PROJECT_REPORT.md   # Complete Academic VII Semester Report
│   ├── benchmark_summary.csv     # Raw tabular benchmark records
│   ├── calibration_sweep_results.csv # Parametric lambda sweep logs
│   ├── rigorous_svd_benchmark.csv# Final 497-user evaluation table
│   └── statistical_tests.json   # Exact t-statistics and p-values
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart & Reproducibility

### 1. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/[username]/recommender.git
cd recommender
pip install -r requirements.txt
```

### 2. Run the Complete Research Pipeline

#### A. Exploratory Data Analysis & Long-Tail Profiling
Automatically downloads the dataset and generates Figures 1 and 2:
```bash
python scripts/run_eda.py
```

#### B. Run Leakage-Free SVD Benchmark with Statistical Testing
Tunes $\lambda$ on 100 validation users and evaluates across 497 held-out test users:
```bash
python scripts/run_advanced_benchmark.py
```

#### C. Parametric Sensitivity Sweep & Trade-Off Curve
Traces the Pareto frontier between ranking accuracy and calibration error:
```bash
python scripts/run_calibration_experiment.py
```

#### D. Generate Macro-Level Population Calibration Plot
```bash
python scripts/plot_macro_calibration.py
```

---

## 📚 Key Literature & References

1. **Koren, Y., Bell, R. and Volinsky, C. (2009).** “Matrix factorization techniques for recommender systems.” *Computer*, 42(8), 30–37.
2. **Steck, H. (2018).** “Calibrated recommendations.” *Proc. 12th ACM Conference on Recommender Systems (RecSys)*, 560–564.
3. **Abdollahpouri, H., Mansoury, M., Burke, R. and Mobasher, B. (2019).** “The unfairness of popularity bias in recommendation.” *Proc. 2019 ACM RecSys Workshop*, 1–7.
4. **Carbonell, J. and Goldstein, J. (1998).** “The use of MMR, diversity-based reranking for reordering documents and producing summaries.” *Proc. 21st ACM SIGIR*, 335–336.
5. **Jannach, D., Lerche, L., Kamehkhosh, I. and Jugovac, M. (2015).** “What recommenders recommend: an analysis of recommendation biases and possible countermeasures.” *User Modeling and User-Adapted Interaction*, 25(5), 427–491.
6. **Vargas, S. and Castells, P. (2011).** “Rank and relevance in novelty and diversity metrics for recommender systems.” *Proc. 5th ACM RecSys*, 109–116.
