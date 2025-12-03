# ===========================================
# Preprocessing Module - Project 1: Optimization
# ===========================================

"""
Module for preprocessing steel plates fault data.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Preprocess data: encode labels, split, and scale.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target labels
    test_size : float
        Test set proportion
    random_state : int
        Random seed
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Preprocessed data
    scaler : StandardScaler
        Fitted scaler
    encoder : LabelEncoder
        Fitted encoder
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
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Data preprocessed:")
    print(f"  - Training: {X_train.shape[0]} samples")
    print(f"  - Test: {X_test.shape[0]} samples")
    print(f"  - Classes: {encoder.classes_.tolist()}")
    
    return X_train, X_test, y_train, y_test, scaler, encoder

