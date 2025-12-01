"""Pipeline de preprocessing complet."""

import pandas as pd
import numpy as np
from pathlib import Path

from .cleaning import load_matches, clean_matches
from .features import build_player_history, create_features

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RANDOM_STATE, N_HISTORICAL_MATCHES, N_SURFACE_MATCHES


class TennisPreprocessor:
    """Preprocesseur pour les données de tennis."""
    
    def __init__(self, n_hist=N_HISTORICAL_MATCHES, n_surf=N_SURFACE_MATCHES):
        self.n_hist = n_hist
        self.n_surf = n_surf
        self.rng = np.random.default_rng(RANDOM_STATE)
        self.history = None
        self.categorical_cols = ["surface", "tourney_level", "round"]
        self.feature_cols = None
        self.fitted = False
    
    def fit_transform(self, matches_df):
        """Fit et transforme le training set."""
        # Construire historique
        self.history = build_player_history(matches_df)
        
        # Créer features
        features_df = create_features(matches_df, self.history, self.n_hist, self.n_surf, self.rng)
        
        # Encoder catégorielles
        features_df = self._encode_categorical(features_df, fit=True)
        
        # Sauvegarder colonnes
        drop_cols = ["target"]
        self.feature_cols = [c for c in features_df.columns if c not in drop_cols]
        self.fitted = True
        
        return features_df
    
    def transform(self, matches_df, all_matches_df):
        """Transforme le test set (utilise all_matches pour l'historique complet)."""
        if not self.fitted:
            raise ValueError("Preprocessor not fitted")
        
        # Reconstruire historique avec toutes les données
        full_history = build_player_history(all_matches_df)
        
        features_df = create_features(matches_df, full_history, self.n_hist, self.n_surf, self.rng)
        features_df = self._encode_categorical(features_df, fit=False)
        
        # Aligner colonnes avec train
        for col in self.feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0
        
        return features_df[[c for c in self.feature_cols if c in features_df.columns] + ["target"]]
    
    def _encode_categorical(self, df, fit=True):
        """One-hot encode les variables catégorielles."""
        if fit:
            self.categories_ = {}
            for col in self.categorical_cols:
                if col in df.columns:
                    self.categories_[col] = df[col].unique().tolist()
        
        for col in self.categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        
        return df
    
    def get_X_y(self, features_df):
        """Sépare features et target."""
        X = features_df.drop(columns=["target"])
        y = features_df["target"]
        return X, y


def temporal_split(df, test_size=0.2):
    """Split temporel basé sur la date."""
    df = df.sort_values("tourney_date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def run_preprocessing(raw_dir, save_dir=None):
    """Pipeline complet de preprocessing."""
    print("Loading data...")
    matches = load_matches(raw_dir)
    print(f"Loaded {len(matches)} matches")
    
    matches = clean_matches(matches)
    print(f"After cleaning: {len(matches)} matches")
    
    train_matches, test_matches = temporal_split(matches)
    print(f"Train: {len(train_matches)}, Test: {len(test_matches)}")
    
    print("Creating features...")
    preprocessor = TennisPreprocessor()
    
    train_features = preprocessor.fit_transform(train_matches)
    print(f"Train features created: {train_features.shape}")
    
    test_features = preprocessor.transform(test_matches, matches)
    print(f"Test features created: {test_features.shape}")
    
    X_train, y_train = preprocessor.get_X_y(train_features)
    X_test, y_test = preprocessor.get_X_y(test_features)
    
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        X_train.to_parquet(save_dir / "X_train.parquet")
        X_test.to_parquet(save_dir / "X_test.parquet")
        y_train.to_frame("target").to_parquet(save_dir / "y_train.parquet")
        y_test.to_frame("target").to_parquet(save_dir / "y_test.parquet")
        print(f"Saved to {save_dir}")
    
    return X_train, X_test, y_train, y_test, preprocessor
