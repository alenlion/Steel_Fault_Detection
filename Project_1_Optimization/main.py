# ===========================================
# Main Script - Project 1: Optimization
# ===========================================

"""
Main script for running optimization experiments.
Compares Grid Search, Random Search, and Bayesian Optimization.
"""

import os
import sys
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_steel_data
from preprocessing import preprocess_data
from optimization import (grid_search, random_search, bayesian_optimization,
                          compare_all_methods, plot_convergence, OPTUNA_AVAILABLE)


def main():
    """Main function to run all experiments."""
    
    print("="*60)
    print("PROJECT 1: OPTIMIZATION")
    print("Comparison of Hyperparameter Optimization Algorithms")
    print("="*60)
    
    # Create directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    
    # Step 1: Load data
    print("\n[1] Loading data...")
    X, y = load_steel_data()
    
    # Step 2: Preprocess data
    print("\n[2] Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, encoder = preprocess_data(X, y)
    
    # Step 3: Run optimization experiments
    print("\n[3] Running optimization experiments...")
    
    all_results = []
    
    # Compare methods for each model
    for model_name in ['SVM', 'RandomForest', 'NeuralNetwork']:
        print(f"\n{'='*40}")
        print(f"Model: {model_name}")
        print('='*40)
        
        comparison = compare_all_methods(X_train, y_train, model_name, cv=5, save_dir='figures')
        comparison['Model'] = model_name
        all_results.append(comparison)
    
    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Save results
    final_results.to_csv('figures/optimization_results.csv', index=False)
    
    # Step 4: Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(final_results.to_string(index=False))
    
    # Find best overall
    best_idx = final_results['Best Score'].idxmax()
    best = final_results.loc[best_idx]
    print(f"\nBest Overall:")
    print(f"  Model: {best['Model']}")
    print(f"  Method: {best['Method']}")
    print(f"  Score: {best['Best Score']:.4f}")
    
    print("\n" + "="*60)
    print("Optimization experiments completed!")
    print("Results saved to 'figures/' directory")
    print("="*60)


if __name__ == "__main__":
    main()

