# ===========================================
# Data Loader Module - Project 3: Data Mining
# ===========================================

"""
Module for loading steel plates fault dataset.
"""

import pandas as pd
import numpy as np
import os
import urllib.request


# Direct download URL for the dataset
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA"


def load_steel_data(save_local=True, data_dir='data/raw'):
    """
    Load steel plates fault dataset.
    """
    
    local_file = os.path.join(data_dir, 'steel_plates_fault.csv')
    
    if os.path.exists(local_file):
        print("Loading data from local file...")
        df = pd.read_csv(local_file)
    else:
        print("Downloading data from UCI Repository...")
        os.makedirs(data_dir, exist_ok=True)
        
        # Download the file
        raw_file = os.path.join(data_dir, 'Faults.NNA')
        urllib.request.urlretrieve(DATASET_URL, raw_file)
        
        # Column names
        feature_names = [
            'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum',
            'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter',
            'Sum_of_Luminosity', 'Minimum_of_Luminosity', 'Maximum_of_Luminosity',
            'Length_of_Conveyer', 'TypeOfSteel_A300', 'TypeOfSteel_A400',
            'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index', 'Square_Index',
            'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index',
            'LogOfAreas', 'Log_X_Index', 'Log_Y_Index',
            'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas',
            'Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'
        ]
        
        # Load data
        df = pd.read_csv(raw_file, sep='\t', header=None, names=feature_names)
        
        # Create class column from one-hot encoded columns
        fault_cols = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
        df['Class'] = df[fault_cols].idxmax(axis=1)
        
        # Drop one-hot columns
        df = df.drop(columns=fault_cols)
        
        # Save processed data
        if save_local:
            df.to_csv(local_file, index=False)
            print(f"Data saved to {local_file}")
    
    # Split features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y


def get_feature_names():
    """Return feature names."""
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
    """Return class names."""
    return ['Bumps', 'Dirtiness', 'K_Scratch', 'Other_Faults', 'Pastry', 'Stains', 'Z_Scratch']

