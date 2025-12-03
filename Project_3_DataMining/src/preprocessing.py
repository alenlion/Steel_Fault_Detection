# ===========================================
# Preprocessing Module - Project 3: Data Mining
# ===========================================

"""
Module for preprocessing data for data mining tasks.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_for_mining(X, y):
    """
    Preprocess data for mining: encode labels and scale features.
    """
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Data preprocessed for mining:")
    print(f"  - Samples: {X_scaled.shape[0]}")
    print(f"  - Features: {X_scaled.shape[1]}")
    print(f"  - Classes: {encoder.classes_.tolist()}")
    
    return X_scaled, y_encoded, scaler, encoder


def compute_statistics(X, feature_names=None):
    """
    Compute descriptive statistics for features.
    """
    
    import pandas as pd
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    if isinstance(X, np.ndarray):
        df = pd.DataFrame(X, columns=feature_names)
    else:
        df = X
    
    stats = df.describe().T
    stats['skewness'] = df.skew()
    stats['kurtosis'] = df.kurtosis()
    
    return stats

