# ===========================================
# Optimization Module - Project 1: Optimization
# ===========================================

"""
Module for hyperparameter optimization.
Compares Grid Search, Random Search, and Bayesian Optimization.
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed.")


# Parameter grids for each model
PARAM_GRIDS = {
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'poly']
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    'NeuralNetwork': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
}


def grid_search(X, y, model_name='SVM', cv=5):
    """
    Perform Grid Search optimization.
    
    Returns dict with best_params, best_score, elapsed_time, cv_results
    """
    
    if model_name == 'SVM':
        model = SVC(probability=True, random_state=42)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    elif model_name == 'NeuralNetwork':
        model = MLPClassifier(max_iter=500, random_state=42)
    
    param_grid = PARAM_GRIDS[model_name]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    print(f"\n[Grid Search] {model_name}...")
    start_time = time.time()
    
    gs = GridSearchCV(model, param_grid, cv=skf, scoring='accuracy', n_jobs=-1, return_train_score=True)
    gs.fit(X, y)
    
    elapsed = time.time() - start_time
    
    print(f"  Best Score: {gs.best_score_:.4f}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Best Params: {gs.best_params_}")
    
    return {
        'method': 'Grid Search',
        'model': model_name,
        'best_params': gs.best_params_,
        'best_score': gs.best_score_,
        'elapsed_time': elapsed,
        'cv_results': pd.DataFrame(gs.cv_results_),
        'best_estimator': gs.best_estimator_
    }


def random_search(X, y, model_name='SVM', n_iter=30, cv=5):
    """
    Perform Random Search optimization.
    
    Returns dict with best_params, best_score, elapsed_time, cv_results
    """
    
    if model_name == 'SVM':
        model = SVC(probability=True, random_state=42)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    elif model_name == 'NeuralNetwork':
        model = MLPClassifier(max_iter=500, random_state=42)
    
    param_dist = PARAM_GRIDS[model_name]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    print(f"\n[Random Search] {model_name}...")
    start_time = time.time()
    
    rs = RandomizedSearchCV(model, param_dist, n_iter=n_iter, cv=skf, 
                            scoring='accuracy', n_jobs=-1, random_state=42,
                            return_train_score=True)
    rs.fit(X, y)
    
    elapsed = time.time() - start_time
    
    print(f"  Best Score: {rs.best_score_:.4f}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Best Params: {rs.best_params_}")
    
    return {
        'method': 'Random Search',
        'model': model_name,
        'best_params': rs.best_params_,
        'best_score': rs.best_score_,
        'elapsed_time': elapsed,
        'cv_results': pd.DataFrame(rs.cv_results_),
        'best_estimator': rs.best_estimator_
    }


def bayesian_optimization(X, y, model_name='SVM', n_trials=30, cv=5):
    """
    Perform Bayesian Optimization using Optuna.
    
    Returns dict with best_params, best_score, elapsed_time, study
    """
    
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not installed. pip install optuna")
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    def objective(trial):
        if model_name == 'SVM':
            C = trial.suggest_float('C', 0.1, 100, log=True)
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto', 0.01, 0.1])
            kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])
            model = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=42)
            
        elif model_name == 'RandomForest':
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 5, 30)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_split=min_samples_split, random_state=42)
            
        elif model_name == 'NeuralNetwork':
            layer1 = trial.suggest_int('layer1', 32, 128)
            layer2 = trial.suggest_int('layer2', 16, 64)
            alpha = trial.suggest_float('alpha', 0.0001, 0.01, log=True)
            model = MLPClassifier(hidden_layer_sizes=(layer1, layer2), alpha=alpha,
                                  max_iter=500, random_state=42)
        
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            scores.append(model.score(X[val_idx], y[val_idx]))
        return np.mean(scores)
    
    print(f"\n[Bayesian Optimization] {model_name}...")
    start_time = time.time()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    elapsed = time.time() - start_time
    
    print(f"  Best Score: {study.best_value:.4f}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Best Params: {study.best_params}")
    
    return {
        'method': 'Bayesian Optimization',
        'model': model_name,
        'best_params': study.best_params,
        'best_score': study.best_value,
        'elapsed_time': elapsed,
        'study': study
    }


def compare_all_methods(X, y, model_name='SVM', cv=5, save_dir='figures'):
    """
    Compare all optimization methods for a given model.
    
    Returns comparison DataFrame and saves plots.
    """
    
    results = []
    
    # Grid Search
    gs = grid_search(X, y, model_name, cv)
    results.append({
        'Method': 'Grid Search',
        'Best Score': gs['best_score'],
        'Time (s)': gs['elapsed_time']
    })
    
    # Random Search
    rs = random_search(X, y, model_name, cv=cv)
    results.append({
        'Method': 'Random Search',
        'Best Score': rs['best_score'],
        'Time (s)': rs['elapsed_time']
    })
    
    # Bayesian Optimization
    if OPTUNA_AVAILABLE:
        bo = bayesian_optimization(X, y, model_name, cv=cv)
        results.append({
            'Method': 'Bayesian Optimization',
            'Best Score': bo['best_score'],
            'Time (s)': bo['elapsed_time']
        })
    
    comparison_df = pd.DataFrame(results)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Score comparison
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    axes[0].bar(comparison_df['Method'], comparison_df['Best Score'], color=colors[:len(comparison_df)])
    axes[0].set_ylabel('Best Score (Accuracy)')
    axes[0].set_title(f'{model_name} - Score Comparison')
    axes[0].set_ylim([comparison_df['Best Score'].min() - 0.05, 1.0])
    for i, v in enumerate(comparison_df['Best Score']):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
    
    # Time comparison
    axes[1].bar(comparison_df['Method'], comparison_df['Time (s)'], color=colors[:len(comparison_df)])
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_title(f'{model_name} - Time Comparison')
    for i, v in enumerate(comparison_df['Time (s)']):
        axes[1].text(i, v + 0.5, f'{v:.1f}s', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/optimization_comparison_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df


def plot_convergence(study, save_path='figures/convergence.png'):
    """
    Plot Optuna optimization convergence.
    """
    
    if not OPTUNA_AVAILABLE:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    trials = [t.number for t in study.trials]
    values = [t.value for t in study.trials]
    best_values = [max(values[:i+1]) for i in range(len(values))]
    
    ax.plot(trials, values, 'o-', alpha=0.5, label='Trial Value')
    ax.plot(trials, best_values, 'r-', linewidth=2, label='Best Value')
    
    ax.set_xlabel('Trial')
    ax.set_ylabel('Accuracy')
    ax.set_title('Bayesian Optimization Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


if __name__ == "__main__":
    print("Optimization module loaded successfully!")
    print(f"Optuna available: {OPTUNA_AVAILABLE}")

