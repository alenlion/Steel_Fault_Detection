# ===========================================
# Data Loader Module - Project 1: Optimization
# ===========================================

"""
Module for loading steel plates fault dataset from OpenML.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import os


def load_steel_data(data_id=1504, save_local=True, data_dir='data/raw'):
    """
    Load steel plates fault dataset from OpenML.
    
    Parameters:
    -----------
    data_id : int
        OpenML dataset ID (default: 1504)
    save_local : bool
        Whether to save data locally
    data_dir : str
        Directory to save/load data
        
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target labels
    """
    
    local_file = os.path.join(data_dir, 'steel_plates_fault.csv')
    
    if os.path.exists(local_file):
        print("Loading data from local file...")
        df = pd.read_csv(local_file)
        X = df.drop('Class', axis=1)
        y = df['Class']
    else:
        print("Downloading data from OpenML...")
        steel = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
        X = steel.data
        y = steel.target
        
        if save_local:
            os.makedirs(data_dir, exist_ok=True)
            df = X.copy()
            df['Class'] = y
            df.to_csv(local_file, index=False)
            print(f"Data saved to {local_file}")
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y


if __name__ == "__main__":
    X, y = load_steel_data()
    print(f"Features: {X.columns.tolist()}")
    print(f"Classes: {y.unique().tolist()}")

