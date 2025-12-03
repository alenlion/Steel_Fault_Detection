# ===========================================
# Preprocessing Module - Project 2: Machine Learning
# ===========================================

"""
Module for preprocessing steel plates fault data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Preprocess data: encode labels, split, and scale.
    """
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Data preprocessed:")
    print(f"  - Training: {X_train_scaled.shape[0]} samples")
    print(f"  - Test: {X_test_scaled.shape[0]} samples")
    print(f"  - Classes: {encoder.classes_.tolist()}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoder


def select_features_kbest(X_train, X_test, y_train, k=15):
    """
    Select top k features using univariate statistical tests.
    """
    
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_indices = selector.get_support(indices=True)
    
    print(f"Selected {k} best features")
    
    return X_train_selected, X_test_selected, selector, selected_indices


def get_feature_importance(X, y, feature_names=None):
    """
    Get feature importance using Random Forest.
    """
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return importance_df, rf

