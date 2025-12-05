"""
Main script pour le projet de prédiction tennis.
Configurer les paramètres dans src/config.py
"""

import pandas as pd
import numpy as np
from src.config import (
    RAW_DIR, PROCESSED_DIR, MODEL_NAME, 
    AUTOML_TIME_BUDGET, SKIP_PREPROCESSING
)
from src.preprocessing.pipeline import run_preprocessing
from src.training.train import train_model, train_with_automl, save_model, evaluate_model


def load_processed_data():
    """Charge les données preprocessées."""
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["target"]
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["target"]
    return X_train, X_test, y_train, y_test


def main():
    print("=" * 50)
    print("TENNIS MATCH PREDICTION")
    print("=" * 50)
    
    # Preprocessing
    if SKIP_PREPROCESSING and (PROCESSED_DIR / "X_train.parquet").exists():
        print("Loading preprocessed data...")
        X_train, X_test, y_train, y_test = load_processed_data()
        print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    else:
        X_train, X_test, y_train, y_test, _ = run_preprocessing(RAW_DIR, save_dir=PROCESSED_DIR)
    
    # Training
    print("\n" + "=" * 50)
    print(f"TRAINING: {MODEL_NAME}")
    print("=" * 50)
    
    if MODEL_NAME == "AutoML":
        model, info = train_with_automl(X_train, y_train, time_budget=AUTOML_TIME_BUDGET)
    else:
        model, info = train_model(X_train, y_train, model_name=MODEL_NAME)
    
    # Évaluation sur test
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    
    test_metrics = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Log Loss: {test_metrics['log_loss']:.4f}")
    print(f"ROC AUC: {test_metrics['roc_auc']:.4f}")
    
    # Baseline
    # Baseline: le mieux classé gagne
    rank_a = X_test["rank_a"].fillna(np.inf)
    rank_b = X_test["rank_b"].fillna(np.inf)
    baseline_pred = (rank_a <= rank_b).astype(int)
    baseline_acc = (baseline_pred.values == y_test.values).mean()
    print(f"\nBaseline (best rank wins): {baseline_acc:.4f}")
    print(f"Improvement: +{(test_metrics['accuracy'] - baseline_acc) * 100:.2f}%")
    
    # Save
    save_model(model)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
