# ATP Tennis Match Prediction

**ENSAE Python Project**
*Authors: Dejean William, Sala Satorre Daniel, Auvepre Edouard*

The final notebook is located at `notebooks/rendu_final.ipynb`.

## Table of Contents

1. [Topic and Research Question](#topic-and-research-question)
2. [Database](#database)
3. [Modeling](#modeling)
4. [Comparison with Bookmakers](#comparison-with-bookmakers)
5. [Main Results](#main-results)
6. [Project Structure](#project-structure)
7. [Installation and Reproducibility](#installation-and-reproducibility)

## Topic and Research Question

The objective of this project is to **predict the winner of a tennis match** using historical ATP circuit data (2000-2025).

Unlike "black box" approaches, we favored an **interpretable statistical approach** based on logistic regression. This allows us to precisely quantify the impact of each variable (ranking, Elo, recent form, surface) on match outcome.

The ultimate goal is to compare our model against bookmaker odds to evaluate its relevance.

## Database

The data comes from the [TML-Database](https://github.com/Tennismylife/TML-Database) repository.

- **Volume:** ~78,000 matches (2000-2025)
- **Cleaning:** Removal of incomplete matches (retirements, walkovers) representing ~3.8% of the dataset

### Feature Engineering

To avoid data leakage, we:
1. **Randomized** player assignment to `Player A` and `Player B`
2. Calculated features using only data **prior** to each match

**Features created:**
- **Ranking:** ATP Rank, ATP Points, Ranking Ratio
- **Elo:** Global and surface-specific Elo score (dynamically updated)
- **Recent Form:** Win rate, service stats with temporal decay (180-day half-life)
- **Context:** Surface win rate, Head-to-Head, tournament level
- **Physical:** Age, height, dominant hand

## Modeling

We used **logistic regression** via `statsmodels` to obtain p-values and interpret the coefficients.

**Significant variables:** Elo, ranking (rank, points), win rate, surface win rate, H2H, service stats, age.

**Non-significant variables:** Height, dominant hand, tournament round.

## Comparison with Bookmakers

We compared our predictions to Bet365 odds on **2025** matches (316 matched matches).

| Metric | Model | Bookmakers |
|--------|-------|------------|
| Accuracy | ~67% | ~72% |
| ROI (all bets) | ~-1.5% | - |

**Conclusion:** Our model has lower accuracy than bookmakers. The negative ROI confirms that it is difficult to beat the market using only public data.

## Main Results

| Model | Accuracy | ROC AUC |
|-------|----------|---------|
| Baseline (better ranked player) | 65.8% | - |
| **Logistic Regression** | **67.7%** | **0.7465** |

- The model beats the baseline by +1.9 points
- **Elo** and **ranking** are the best predictors
- Physical variables (height, hand) are not significant

## Project Structure
```
├── config.py                     # Parameters and paths
├── src/
│   ├── preprocessing/
│   │   ├── cleaning.py           # Loading and cleaning
│   │   ├── features.py           # Feature engineering
│   │   └── pipeline.py           # TennisPreprocessor class
│   ├── training/
│   │   └── models.py             # StatsLogitClassifier
│   └── evaluation/
│       ├── bookmakers.py         # Odds loading
│       └── comparison.py         # ROI, value bets calculations
├── notebooks/
│   └── rendu_final.ipynb         # Main notebook
├── data/
│   └── raw/                      # CSVs (downloaded automatically)
└── requirements.txt
```

## Installation and Reproducibility

```bash
# Clone the project
git clone https://github.com/Edouard386/Projet-python.git

# Navigate to the folder
cd Projet-python

# Install dependencies
pip install -r requirements.txt
```
Then run `notebooks/rendu_final.ipynb`. Data is downloaded automatically on first launch.
