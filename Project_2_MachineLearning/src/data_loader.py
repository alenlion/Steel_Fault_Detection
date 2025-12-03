# ===========================================
# Data Loader Module - Project 2: Machine Learning
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


def get_feature_names():
    """Return feature names for steel plates fault dataset."""
    return [
        'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum',
        'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter',
        'Sum_of_Luminosity', 'Minimum_of_Luminosity', 'Maximum_of_Luminosity',
        'Length_of_Conveyer', 'TypeOfSteel_A300', 'TypeOfSteel_A400',
        'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index', 'Square_Index',
        'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index',
        'LogOfAreas', 'Log_X_Index', 'Log_Y_Index',
        'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas'
    ]


def get_class_names():
    """Return class names for steel plates fault dataset."""
    return ['Bumps', 'Dirtiness', 'K_Scratch', 'Other_Faults', 'Pastry', 'Stains', 'Z_Scratch']

