# ===========================================
# Main Script - Project 2: Machine Learning
# ===========================================

"""
Main script for running machine learning experiments.
Compares multiple classification algorithms.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_steel_data, get_class_names, get_feature_names
from preprocessing import preprocess_data, get_feature_importance
from models import get_models, compare_models, get_classification_report_str, get_confusion_matrix
from visualization import (plot_model_comparison, plot_confusion_matrix, 
                           plot_roc_curves, plot_feature_importance, plot_metrics_radar)


def main():
    """Main function to run all experiments."""
    
    print("="*60)
    print("PROJECT 2: MACHINE LEARNING & PATTERN RECOGNITION")
    print("Multi-class Classification of Steel Plate Faults")
    print("="*60)
    
    # Create directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    
    # Step 1: Load data
    print("\n[1] Loading data...")
    X, y = load_steel_data()
    feature_names = get_feature_names()
    class_names = get_class_names()
    
    # Step 2: Preprocess data
    print("\n[2] Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, encoder = preprocess_data(X, y)
    
    # Step 3: Compare models
    print("\n[3] Training and comparing models...")
    results_df, trained_models = compare_models(X_train, X_test, y_train, y_test)
    
    # Save results
    results_df.to_csv('figures/model_comparison_results.csv', index=False)
    
    # Print results
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Step 4: Visualizations
    print("\n[4] Creating visualizations...")
    
    # Model comparison
    plot_model_comparison(results_df, metric='Accuracy', save_path='figures/model_comparison_accuracy.png')
    plot_model_comparison(results_df, metric='F1-Score', save_path='figures/model_comparison_f1.png')
    
    # Best model analysis
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    
    print(f"\n[5] Analyzing best model: {best_model_name}")
    
    # Confusion matrix
    cm = get_confusion_matrix(best_model, X_test, y_test)
    plot_confusion_matrix(cm, class_names=encoder.classes_, normalize=False,
                         save_path='figures/confusion_matrix.png')
    plot_confusion_matrix(cm, class_names=encoder.classes_, normalize=True,
                         save_path='figures/confusion_matrix_normalized.png')
    
    # ROC curves
    plot_roc_curves(trained_models, X_test, y_test, save_path='figures/roc_curves.png')
    
    # Feature importance
    print("\n[6] Computing feature importance...")
    importance_df, _ = get_feature_importance(X_train, y_train, feature_names)
    importance_df.to_csv('figures/feature_importance.csv', index=False)
    plot_feature_importance(importance_df, top_n=15, save_path='figures/feature_importance.png')
    
    # Classification report for best model
    print(f"\n[7] Classification Report for {best_model_name}:")
    print("-"*60)
    report = get_classification_report_str(best_model, X_test, y_test, target_names=encoder.classes_)
    print(report)
    
    # Save report
    with open('figures/classification_report.txt', 'w') as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write("="*60 + "\n")
        f.write(report)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best Model: {best_model_name}")
    print(f"Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
    print(f"F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")
    print(f"Top 5 Important Features:")
    for i, row in importance_df.head(5).iterrows():
        print(f"  - {row['Feature']}: {row['Importance']:.4f}")
    
    print("\n" + "="*60)
    print("Machine Learning experiments completed!")
    print("Results saved to 'figures/' directory")
    print("="*60)


if __name__ == "__main__":
    main()

