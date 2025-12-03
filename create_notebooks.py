"""
Script to create comprehensive Jupyter notebooks for all three projects.
Creates both English and Turkish versions.
"""

import json
import os

def create_notebook(cells):
    """Create a Jupyter notebook structure."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def markdown_cell(content):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.split('\n')
    }

def code_cell(content):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content.split('\n')
    }

# ============================================================
# PROJECT 1: OPTIMIZATION - ENGLISH
# ============================================================
def create_project1_en():
    cells = [
        markdown_cell("""# Project 1: Hyperparameter Optimization

**Course**: Optimization Algorithms

**Dataset**: Steel Plates Fault Detection (UCI Machine Learning Repository)

**Objective**: Compare hyperparameter optimization methods

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Setup and Data Loading](#2-setup-and-data-loading)
3. [Data Exploration](#3-data-exploration)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Grid Search Optimization](#5-grid-search-optimization)
6. [Random Search Optimization](#6-random-search-optimization)
7. [Bayesian Optimization](#7-bayesian-optimization)
8. [Comparison and Analysis](#8-comparison-and-analysis)
9. [Summary and Conclusions](#9-summary-and-conclusions)"""),

        markdown_cell("""---
## 1. Introduction

### What is Hyperparameter Optimization?

Hyperparameter optimization is the process of finding the best configuration for a machine learning model. Unlike model parameters (learned during training), hyperparameters are set before training begins.

### Methods Compared:

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Grid Search** | Exhaustively searches all combinations | Thorough, reproducible | Slow, expensive |
| **Random Search** | Randomly samples combinations | Faster, efficient | May miss optimal |
| **Bayesian Optimization** | Uses probabilistic model | Smart sampling, efficient | More complex |

### Models to Optimize:
- Support Vector Machine (SVM)
- Random Forest
- Neural Network (MLP)"""),

        markdown_cell("""---
## 2. Setup and Data Loading"""),

        code_cell("""# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Optuna for Bayesian Optimization
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
    print("✅ Optuna is available for Bayesian Optimization")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not installed. Install with: pip install optuna")

# Visualization settings
%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')
sns.set_palette('husl')

# Set random seed
np.random.seed(42)

print("\\n✅ All libraries imported successfully!")"""),

        code_cell("""# Load the Steel Plates Fault dataset
data_path = '../data/raw/steel_plates_fault.csv'

df = pd.read_csv(data_path)

print("✅ Data loaded successfully!")
print(f"\\n📊 Dataset Shape: {df.shape[0]:,} samples × {df.shape[1]} columns")"""),

        markdown_cell("""---
## 3. Data Exploration"""),

        code_cell("""# Display first few rows
print("📋 First 5 rows of the dataset:")
df.head()"""),

        code_cell("""# Dataset information
print("ℹ️ Dataset Information:")
print("=" * 50)
df.info()"""),

        code_cell("""# Statistical summary
print("📊 Statistical Summary:")
df.describe()"""),

        code_cell("""# Check for missing values
missing = df.isnull().sum().sum()
print(f"🔍 Missing Values: {missing}")
if missing == 0:
    print("✅ No missing values found!")"""),

        code_cell("""# Target variable distribution
print("🎯 Target Variable Distribution:")
print("=" * 50)

class_counts = df['Class'].value_counts()
print(class_counts)

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette('husl', len(class_counts))
bars = ax.bar(class_counts.index, class_counts.values, color=colors, edgecolor='black')
ax.set_xlabel('Fault Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Steel Plate Fault Types', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')

for bar, count in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
            f'{count}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""---
## 4. Data Preprocessing"""),

        code_cell("""# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

print(f"📊 Features Shape: {X.shape}")
print(f"🎯 Target Shape: {y.shape}")
print(f"\\n📋 Feature Names: {X.columns.tolist()}")"""),

        code_cell("""# Encode target labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("✅ Labels encoded!")
print(f"\\n🏷️ Classes: {encoder.classes_.tolist()}")"""),

        code_cell("""# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("✅ Data split completed!")
print(f"\\n📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")"""),

        code_cell("""# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features scaled using StandardScaler!")"""),

        markdown_cell("""---
## 5. Grid Search Optimization

### What is Grid Search?

Grid Search exhaustively evaluates all possible combinations of hyperparameters from a predefined grid.

**Algorithm:**
1. Define parameter grid
2. For each combination: Train model and evaluate using cross-validation
3. Return best combination"""),

        code_cell("""# Define parameter grids
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

print("📋 Parameter Grids Defined:")
for model, params in PARAM_GRIDS.items():
    total = np.prod([len(v) for v in params.values()])
    print(f"\\n  {model}: {total} combinations")"""),

        code_cell("""def perform_grid_search(X, y, model_name, cv=5):
    \"\"\"Perform Grid Search optimization.\"\"\"
    if model_name == 'SVM':
        model = SVC(probability=True, random_state=42)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    elif model_name == 'NeuralNetwork':
        model = MLPClassifier(max_iter=500, random_state=42)
    
    param_grid = PARAM_GRIDS[model_name]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    print(f"\\n🔍 Grid Search for {model_name}...")
    start_time = time.time()
    
    gs = GridSearchCV(model, param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
    gs.fit(X, y)
    
    elapsed = time.time() - start_time
    
    print(f"  ✅ Best Score: {gs.best_score_:.4f}")
    print(f"  ⏱️ Time: {elapsed:.2f} seconds")
    print(f"  🎯 Best Parameters: {gs.best_params_}")
    
    return {'method': 'Grid Search', 'model': model_name, 
            'best_score': gs.best_score_, 'elapsed_time': elapsed,
            'best_params': gs.best_params_, 'best_estimator': gs.best_estimator_}"""),

        code_cell("""# Run Grid Search for all models
print("="*60)
print("GRID SEARCH OPTIMIZATION")
print("="*60)

grid_results = {}
for model_name in ['SVM', 'RandomForest', 'NeuralNetwork']:
    grid_results[model_name] = perform_grid_search(X_train_scaled, y_train, model_name)"""),

        markdown_cell("""---
## 6. Random Search Optimization

### What is Random Search?

Random Search randomly samples hyperparameter combinations. Research shows it can find good solutions faster than Grid Search.

**Advantages:**
- Explores more of the hyperparameter space
- More efficient when some parameters are more important
- Faster with large parameter spaces"""),

        code_cell("""def perform_random_search(X, y, model_name, n_iter=30, cv=5):
    \"\"\"Perform Random Search optimization.\"\"\"
    if model_name == 'SVM':
        model = SVC(probability=True, random_state=42)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    elif model_name == 'NeuralNetwork':
        model = MLPClassifier(max_iter=500, random_state=42)
    
    param_dist = PARAM_GRIDS[model_name]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    print(f"\\n🎲 Random Search for {model_name}...")
    start_time = time.time()
    
    rs = RandomizedSearchCV(model, param_dist, n_iter=n_iter, cv=skf, 
                            scoring='accuracy', n_jobs=-1, random_state=42)
    rs.fit(X, y)
    
    elapsed = time.time() - start_time
    
    print(f"  ✅ Best Score: {rs.best_score_:.4f}")
    print(f"  ⏱️ Time: {elapsed:.2f} seconds")
    print(f"  🎯 Best Parameters: {rs.best_params_}")
    
    return {'method': 'Random Search', 'model': model_name,
            'best_score': rs.best_score_, 'elapsed_time': elapsed,
            'best_params': rs.best_params_, 'best_estimator': rs.best_estimator_}"""),

        code_cell("""# Run Random Search for all models
print("="*60)
print("RANDOM SEARCH OPTIMIZATION")
print("="*60)

random_results = {}
for model_name in ['SVM', 'RandomForest', 'NeuralNetwork']:
    random_results[model_name] = perform_random_search(X_train_scaled, y_train, model_name)"""),

        markdown_cell("""---
## 7. Bayesian Optimization

### What is Bayesian Optimization?

Bayesian Optimization uses a probabilistic model to intelligently select next evaluation points based on:
1. **Exploration**: Trying new, uncertain regions
2. **Exploitation**: Focusing on promising regions

**Advantages:**
- More efficient than random/grid search
- Uses past evaluations to guide future choices"""),

        code_cell("""def perform_bayesian_optimization(X, y, model_name, n_trials=30, cv=5):
    \"\"\"Perform Bayesian Optimization using Optuna.\"\"\"
    if not OPTUNA_AVAILABLE:
        print("⚠️ Optuna not available.")
        return None
    
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
    
    print(f"\\n🧠 Bayesian Optimization for {model_name}...")
    start_time = time.time()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    elapsed = time.time() - start_time
    
    print(f"  ✅ Best Score: {study.best_value:.4f}")
    print(f"  ⏱️ Time: {elapsed:.2f} seconds")
    print(f"  🎯 Best Parameters: {study.best_params}")
    
    return {'method': 'Bayesian Optimization', 'model': model_name,
            'best_score': study.best_value, 'elapsed_time': elapsed,
            'best_params': study.best_params, 'study': study}"""),

        code_cell("""# Run Bayesian Optimization for all models
print("="*60)
print("BAYESIAN OPTIMIZATION")
print("="*60)

bayesian_results = {}
if OPTUNA_AVAILABLE:
    for model_name in ['SVM', 'RandomForest', 'NeuralNetwork']:
        bayesian_results[model_name] = perform_bayesian_optimization(X_train_scaled, y_train, model_name)
else:
    print("⚠️ Skipping Bayesian Optimization (Optuna not installed)")"""),

        markdown_cell("""---
## 8. Comparison and Analysis"""),

        code_cell("""# Compile all results
all_results = []

for model_name in ['SVM', 'RandomForest', 'NeuralNetwork']:
    if model_name in grid_results:
        all_results.append({
            'Model': model_name, 'Method': 'Grid Search',
            'Best Score': grid_results[model_name]['best_score'],
            'Time (s)': grid_results[model_name]['elapsed_time']
        })
    
    if model_name in random_results:
        all_results.append({
            'Model': model_name, 'Method': 'Random Search',
            'Best Score': random_results[model_name]['best_score'],
            'Time (s)': random_results[model_name]['elapsed_time']
        })
    
    if model_name in bayesian_results and bayesian_results[model_name]:
        all_results.append({
            'Model': model_name, 'Method': 'Bayesian Optimization',
            'Best Score': bayesian_results[model_name]['best_score'],
            'Time (s)': bayesian_results[model_name]['elapsed_time']
        })

results_df = pd.DataFrame(all_results)

print("📊 OPTIMIZATION RESULTS SUMMARY")
print("="*70)
print(results_df.to_string(index=False))
print("="*70)"""),

        code_cell("""# Visualize comparison for each model
models = ['SVM', 'RandomForest', 'NeuralNetwork']

for model_name in models:
    model_df = results_df[results_df['Model'] == model_name]
    if len(model_df) == 0:
        continue
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} - Optimization Methods Comparison', fontsize=14, fontweight='bold')
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Score comparison
    bars1 = axes[0].bar(model_df['Method'], model_df['Best Score'], color=colors[:len(model_df)], edgecolor='black')
    axes[0].set_ylabel('Best Score (Accuracy)', fontweight='bold')
    axes[0].set_title('Score Comparison', fontweight='bold')
    axes[0].set_ylim([model_df['Best Score'].min() - 0.05, 1.0])
    for bar, score in zip(bars1, model_df['Best Score']):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.4f}', ha='center', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)
    
    # Time comparison
    bars2 = axes[1].bar(model_df['Method'], model_df['Time (s)'], color=colors[:len(model_df)], edgecolor='black')
    axes[1].set_ylabel('Time (seconds)', fontweight='bold')
    axes[1].set_title('Time Comparison', fontweight='bold')
    for bar, t in zip(bars2, model_df['Time (s)']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{t:.1f}s', ha='center', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(f'../figures/optimization_comparison_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()"""),

        code_cell("""# Find best overall configuration
best_idx = results_df['Best Score'].idxmax()
best_result = results_df.loc[best_idx]

print("🏆 BEST OVERALL CONFIGURATION")
print("="*50)
print(f"Model: {best_result['Model']}")
print(f"Method: {best_result['Method']}")
print(f"Best Score: {best_result['Best Score']:.4f}")
print(f"Time: {best_result['Time (s)']:.2f} seconds")

# Save results
results_df.to_csv('../figures/optimization_results.csv', index=False)
print("\\n✅ Results saved to '../figures/optimization_results.csv'")"""),

        markdown_cell("""---
## 9. Summary and Conclusions

### Key Findings

| Method | Characteristics |
|--------|-----------------|
| **Grid Search** | Exhaustive, slow but thorough |
| **Random Search** | Fast, good coverage |
| **Bayesian Optimization** | Smart, efficient for expensive evaluations |

### Recommendations

| Scenario | Recommended Method |
|----------|-------------------|
| Small parameter space | Grid Search |
| Large parameter space | Random Search or Bayesian |
| Limited computational budget | Bayesian Optimization |
| Need guaranteed coverage | Grid Search |"""),

        code_cell("""print("="*60)
print("🎉 PROJECT 1: OPTIMIZATION - COMPLETE!")
print("="*60)
print("\\n📁 Output files saved in '../figures/':")
print("  - optimization_comparison_SVM.png")
print("  - optimization_comparison_RandomForest.png")
print("  - optimization_comparison_NeuralNetwork.png")
print("  - optimization_results.csv")
print("\\n✅ All experiments completed successfully!")""")
    ]
    return create_notebook(cells)


# ============================================================
# PROJECT 1: OPTIMIZATION - TURKISH
# ============================================================
def create_project1_tr():
    cells = [
        markdown_cell("""# Proje 1: Hiperparametre Optimizasyonu

**Ders**: Optimizasyon Algoritmaları

**Veri Seti**: Çelik Levha Hata Tespiti (UCI Makine Öğrenimi Deposu)

**Amaç**: Hiperparametre optimizasyon yöntemlerini karşılaştırma

---

## İçindekiler

1. [Giriş](#1-giriş)
2. [Kurulum ve Veri Yükleme](#2-kurulum-ve-veri-yükleme)
3. [Veri Keşfi](#3-veri-keşfi)
4. [Veri Ön İşleme](#4-veri-ön-işleme)
5. [Grid Search Optimizasyonu](#5-grid-search-optimizasyonu)
6. [Random Search Optimizasyonu](#6-random-search-optimizasyonu)
7. [Bayesian Optimizasyonu](#7-bayesian-optimizasyonu)
8. [Karşılaştırma ve Analiz](#8-karşılaştırma-ve-analiz)
9. [Özet ve Sonuçlar](#9-özet-ve-sonuçlar)"""),

        markdown_cell("""---
## 1. Giriş

### Hiperparametre Optimizasyonu Nedir?

Hiperparametre optimizasyonu, bir makine öğrenimi modeli için en iyi yapılandırmayı bulma sürecidir. Model parametrelerinden (eğitim sırasında öğrenilen) farklı olarak, hiperparametreler eğitim başlamadan önce ayarlanır.

### Karşılaştırılan Yöntemler:

| Yöntem | Açıklama | Artıları | Eksileri |
|--------|----------|----------|----------|
| **Grid Search** | Tüm kombinasyonları değerlendirir | Kapsamlı, tekrarlanabilir | Yavaş, pahalı |
| **Random Search** | Rastgele örnekler seçer | Hızlı, verimli | Optimali kaçırabilir |
| **Bayesian Optimizasyon** | Olasılıksal model kullanır | Akıllı örnekleme | Daha karmaşık |

### Optimize Edilecek Modeller:
- Destek Vektör Makinesi (SVM)
- Random Forest
- Yapay Sinir Ağı (MLP)"""),

        markdown_cell("""---
## 2. Kurulum ve Veri Yükleme"""),

        code_cell("""# Gerekli kütüphaneleri içe aktar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn kütüphaneleri
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Bayesian Optimizasyon için Optuna
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
    print("✅ Optuna Bayesian Optimizasyon için kullanılabilir")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna yüklü değil. Yüklemek için: pip install optuna")

# Görselleştirme ayarları
%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')
sns.set_palette('husl')

# Rastgele sayı üreteci için seed
np.random.seed(42)

print("\\n✅ Tüm kütüphaneler başarıyla içe aktarıldı!")"""),

        code_cell("""# Çelik Levha Hata veri setini yükle
veri_yolu = '../data/raw/steel_plates_fault.csv'

df = pd.read_csv(veri_yolu)

print("✅ Veri başarıyla yüklendi!")
print(f"\\n📊 Veri Seti Boyutu: {df.shape[0]:,} örnek × {df.shape[1]} sütun")"""),

        markdown_cell("""---
## 3. Veri Keşfi"""),

        code_cell("""# İlk birkaç satırı göster
print("📋 Veri setinin ilk 5 satırı:")
df.head()"""),

        code_cell("""# Veri seti bilgileri
print("ℹ️ Veri Seti Bilgileri:")
print("=" * 50)
df.info()"""),

        code_cell("""# İstatistiksel özet
print("📊 İstatistiksel Özet:")
df.describe()"""),

        code_cell("""# Eksik değerleri kontrol et
eksik = df.isnull().sum().sum()
print(f"🔍 Eksik Değerler: {eksik}")
if eksik == 0:
    print("✅ Eksik değer bulunamadı!")"""),

        code_cell("""# Hedef değişken dağılımı
print("🎯 Hedef Değişken Dağılımı:")
print("=" * 50)

sinif_sayilari = df['Class'].value_counts()
print(sinif_sayilari)

# Görselleştir
fig, ax = plt.subplots(figsize=(10, 6))
renkler = sns.color_palette('husl', len(sinif_sayilari))
cubuklar = ax.bar(sinif_sayilari.index, sinif_sayilari.values, color=renkler, edgecolor='black')
ax.set_xlabel('Hata Türü', fontsize=12, fontweight='bold')
ax.set_ylabel('Sayı', fontsize=12, fontweight='bold')
ax.set_title('Çelik Levha Hata Türleri Dağılımı', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')

for cubuk, sayi in zip(cubuklar, sinif_sayilari.values):
    ax.text(cubuk.get_x() + cubuk.get_width()/2, cubuk.get_height() + 5, 
            f'{sayi}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/sinif_dagilimi.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""---
## 4. Veri Ön İşleme"""),

        code_cell("""# Özellikleri ve hedefi ayır
X = df.drop('Class', axis=1)
y = df['Class']

print(f"📊 Özellik Boyutu: {X.shape}")
print(f"🎯 Hedef Boyutu: {y.shape}")
print(f"\\n📋 Özellik Adları: {X.columns.tolist()}")"""),

        code_cell("""# Hedef etiketlerini kodla
kodlayici = LabelEncoder()
y_kodlanmis = kodlayici.fit_transform(y)

print("✅ Etiketler kodlandı!")
print(f"\\n🏷️ Sınıflar: {kodlayici.classes_.tolist()}")"""),

        code_cell("""# Veriyi eğitim ve test setlerine ayır
X_egitim, X_test, y_egitim, y_test = train_test_split(
    X, y_kodlanmis, test_size=0.2, random_state=42, stratify=y_kodlanmis
)

print("✅ Veri bölme işlemi tamamlandı!")
print(f"\\n📊 Eğitim seti: {X_egitim.shape[0]} örnek")
print(f"📊 Test seti: {X_test.shape[0]} örnek")"""),

        code_cell("""# Özellikleri ölçeklendir
olcekleyici = StandardScaler()
X_egitim_olcekli = olcekleyici.fit_transform(X_egitim)
X_test_olcekli = olcekleyici.transform(X_test)

print("✅ Özellikler StandardScaler ile ölçeklendirildi!")"""),

        markdown_cell("""---
## 5. Grid Search Optimizasyonu

### Grid Search Nedir?

Grid Search, önceden tanımlanmış bir ızgaradan tüm olası hiperparametre kombinasyonlarını değerlendirir.

**Algoritma:**
1. Parametre ızgarası tanımla
2. Her kombinasyon için: Modeli eğit ve çapraz doğrulama ile değerlendir
3. En iyi kombinasyonu döndür"""),

        code_cell("""# Parametre ızgaralarını tanımla
PARAMETRE_IZGARALARI = {
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

print("📋 Parametre Izgaraları Tanımlandı:")
for model, parametreler in PARAMETRE_IZGARALARI.items():
    toplam = np.prod([len(v) for v in parametreler.values()])
    print(f"\\n  {model}: {toplam} kombinasyon")"""),

        code_cell("""def grid_search_yap(X, y, model_adi, cv=5):
    \"\"\"Grid Search optimizasyonu yap.\"\"\"
    if model_adi == 'SVM':
        model = SVC(probability=True, random_state=42)
    elif model_adi == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    elif model_adi == 'NeuralNetwork':
        model = MLPClassifier(max_iter=500, random_state=42)
    
    param_grid = PARAMETRE_IZGARALARI[model_adi]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    print(f"\\n🔍 {model_adi} için Grid Search...")
    baslangic = time.time()
    
    gs = GridSearchCV(model, param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
    gs.fit(X, y)
    
    gecen_sure = time.time() - baslangic
    
    print(f"  ✅ En İyi Skor: {gs.best_score_:.4f}")
    print(f"  ⏱️ Süre: {gecen_sure:.2f} saniye")
    print(f"  🎯 En İyi Parametreler: {gs.best_params_}")
    
    return {'yontem': 'Grid Search', 'model': model_adi, 
            'en_iyi_skor': gs.best_score_, 'gecen_sure': gecen_sure,
            'en_iyi_parametreler': gs.best_params_, 'en_iyi_model': gs.best_estimator_}"""),

        code_cell("""# Tüm modeller için Grid Search çalıştır
print("="*60)
print("GRID SEARCH OPTİMİZASYONU")
print("="*60)

grid_sonuclari = {}
for model_adi in ['SVM', 'RandomForest', 'NeuralNetwork']:
    grid_sonuclari[model_adi] = grid_search_yap(X_egitim_olcekli, y_egitim, model_adi)"""),

        markdown_cell("""---
## 6. Random Search Optimizasyonu

### Random Search Nedir?

Random Search, hiperparametre kombinasyonlarını rastgele örnekler. Araştırmalar, Grid Search'ten daha hızlı iyi çözümler bulabildiğini göstermektedir.

**Avantajları:**
- Hiperparametre uzayının daha fazlasını keşfeder
- Bazı parametreler daha önemli olduğunda daha verimli
- Büyük parametre uzaylarında daha hızlı"""),

        code_cell("""def random_search_yap(X, y, model_adi, n_iter=30, cv=5):
    \"\"\"Random Search optimizasyonu yap.\"\"\"
    if model_adi == 'SVM':
        model = SVC(probability=True, random_state=42)
    elif model_adi == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    elif model_adi == 'NeuralNetwork':
        model = MLPClassifier(max_iter=500, random_state=42)
    
    param_dist = PARAMETRE_IZGARALARI[model_adi]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    print(f"\\n🎲 {model_adi} için Random Search...")
    baslangic = time.time()
    
    rs = RandomizedSearchCV(model, param_dist, n_iter=n_iter, cv=skf, 
                            scoring='accuracy', n_jobs=-1, random_state=42)
    rs.fit(X, y)
    
    gecen_sure = time.time() - baslangic
    
    print(f"  ✅ En İyi Skor: {rs.best_score_:.4f}")
    print(f"  ⏱️ Süre: {gecen_sure:.2f} saniye")
    print(f"  🎯 En İyi Parametreler: {rs.best_params_}")
    
    return {'yontem': 'Random Search', 'model': model_adi,
            'en_iyi_skor': rs.best_score_, 'gecen_sure': gecen_sure,
            'en_iyi_parametreler': rs.best_params_, 'en_iyi_model': rs.best_estimator_}"""),

        code_cell("""# Tüm modeller için Random Search çalıştır
print("="*60)
print("RANDOM SEARCH OPTİMİZASYONU")
print("="*60)

random_sonuclari = {}
for model_adi in ['SVM', 'RandomForest', 'NeuralNetwork']:
    random_sonuclari[model_adi] = random_search_yap(X_egitim_olcekli, y_egitim, model_adi)"""),

        markdown_cell("""---
## 7. Bayesian Optimizasyonu

### Bayesian Optimizasyon Nedir?

Bayesian Optimizasyon, değerlendirme noktalarını akıllıca seçmek için olasılıksal model kullanır:
1. **Keşif**: Yeni, belirsiz bölgeleri dener
2. **Sömürü**: Umut vadeden bölgelere odaklanır

**Avantajları:**
- Random/Grid Search'ten daha verimli
- Gelecekteki seçimlere rehberlik etmek için geçmiş değerlendirmeleri kullanır"""),

        code_cell("""def bayesian_optimizasyon_yap(X, y, model_adi, n_trials=30, cv=5):
    \"\"\"Optuna kullanarak Bayesian Optimizasyon yap.\"\"\"
    if not OPTUNA_AVAILABLE:
        print("⚠️ Optuna kullanılamıyor.")
        return None
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    def hedef(trial):
        if model_adi == 'SVM':
            C = trial.suggest_float('C', 0.1, 100, log=True)
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto', 0.01, 0.1])
            kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])
            model = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=42)
        elif model_adi == 'RandomForest':
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 5, 30)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_split=min_samples_split, random_state=42)
        elif model_adi == 'NeuralNetwork':
            layer1 = trial.suggest_int('layer1', 32, 128)
            layer2 = trial.suggest_int('layer2', 16, 64)
            alpha = trial.suggest_float('alpha', 0.0001, 0.01, log=True)
            model = MLPClassifier(hidden_layer_sizes=(layer1, layer2), alpha=alpha,
                                  max_iter=500, random_state=42)
        
        skorlar = []
        for egitim_idx, dogrulama_idx in skf.split(X, y):
            model.fit(X[egitim_idx], y[egitim_idx])
            skorlar.append(model.score(X[dogrulama_idx], y[dogrulama_idx]))
        return np.mean(skorlar)
    
    print(f"\\n🧠 {model_adi} için Bayesian Optimizasyon...")
    baslangic = time.time()
    
    calisma = optuna.create_study(direction='maximize')
    calisma.optimize(hedef, n_trials=n_trials, show_progress_bar=True)
    
    gecen_sure = time.time() - baslangic
    
    print(f"  ✅ En İyi Skor: {calisma.best_value:.4f}")
    print(f"  ⏱️ Süre: {gecen_sure:.2f} saniye")
    print(f"  🎯 En İyi Parametreler: {calisma.best_params}")
    
    return {'yontem': 'Bayesian Optimizasyon', 'model': model_adi,
            'en_iyi_skor': calisma.best_value, 'gecen_sure': gecen_sure,
            'en_iyi_parametreler': calisma.best_params, 'calisma': calisma}"""),

        code_cell("""# Tüm modeller için Bayesian Optimizasyon çalıştır
print("="*60)
print("BAYESIAN OPTİMİZASYON")
print("="*60)

bayesian_sonuclari = {}
if OPTUNA_AVAILABLE:
    for model_adi in ['SVM', 'RandomForest', 'NeuralNetwork']:
        bayesian_sonuclari[model_adi] = bayesian_optimizasyon_yap(X_egitim_olcekli, y_egitim, model_adi)
else:
    print("⚠️ Bayesian Optimizasyon atlanıyor (Optuna yüklü değil)")"""),

        markdown_cell("""---
## 8. Karşılaştırma ve Analiz"""),

        code_cell("""# Tüm sonuçları derle
tum_sonuclar = []

for model_adi in ['SVM', 'RandomForest', 'NeuralNetwork']:
    if model_adi in grid_sonuclari:
        tum_sonuclar.append({
            'Model': model_adi, 'Yöntem': 'Grid Search',
            'En İyi Skor': grid_sonuclari[model_adi]['en_iyi_skor'],
            'Süre (s)': grid_sonuclari[model_adi]['gecen_sure']
        })
    
    if model_adi in random_sonuclari:
        tum_sonuclar.append({
            'Model': model_adi, 'Yöntem': 'Random Search',
            'En İyi Skor': random_sonuclari[model_adi]['en_iyi_skor'],
            'Süre (s)': random_sonuclari[model_adi]['gecen_sure']
        })
    
    if model_adi in bayesian_sonuclari and bayesian_sonuclari[model_adi]:
        tum_sonuclar.append({
            'Model': model_adi, 'Yöntem': 'Bayesian Optimizasyon',
            'En İyi Skor': bayesian_sonuclari[model_adi]['en_iyi_skor'],
            'Süre (s)': bayesian_sonuclari[model_adi]['gecen_sure']
        })

sonuclar_df = pd.DataFrame(tum_sonuclar)

print("📊 OPTİMİZASYON SONUÇLARI ÖZETİ")
print("="*70)
print(sonuclar_df.to_string(index=False))
print("="*70)"""),

        code_cell("""# Her model için karşılaştırmayı görselleştir
modeller = ['SVM', 'RandomForest', 'NeuralNetwork']

for model_adi in modeller:
    model_df = sonuclar_df[sonuclar_df['Model'] == model_adi]
    if len(model_df) == 0:
        continue
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_adi} - Optimizasyon Yöntemleri Karşılaştırması', fontsize=14, fontweight='bold')
    
    renkler = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Skor karşılaştırması
    cubuklar1 = axes[0].bar(model_df['Yöntem'], model_df['En İyi Skor'], color=renkler[:len(model_df)], edgecolor='black')
    axes[0].set_ylabel('En İyi Skor (Doğruluk)', fontweight='bold')
    axes[0].set_title('Skor Karşılaştırması', fontweight='bold')
    axes[0].set_ylim([model_df['En İyi Skor'].min() - 0.05, 1.0])
    for cubuk, skor in zip(cubuklar1, model_df['En İyi Skor']):
        axes[0].text(cubuk.get_x() + cubuk.get_width()/2, cubuk.get_height() + 0.01, f'{skor:.4f}', ha='center', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)
    
    # Süre karşılaştırması
    cubuklar2 = axes[1].bar(model_df['Yöntem'], model_df['Süre (s)'], color=renkler[:len(model_df)], edgecolor='black')
    axes[1].set_ylabel('Süre (saniye)', fontweight='bold')
    axes[1].set_title('Süre Karşılaştırması', fontweight='bold')
    for cubuk, t in zip(cubuklar2, model_df['Süre (s)']):
        axes[1].text(cubuk.get_x() + cubuk.get_width()/2, cubuk.get_height() + 0.5, f'{t:.1f}s', ha='center', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(f'../figures/optimizasyon_karsilastirma_{model_adi}.png', dpi=300, bbox_inches='tight')
    plt.show()"""),

        code_cell("""# En iyi genel yapılandırmayı bul
en_iyi_idx = sonuclar_df['En İyi Skor'].idxmax()
en_iyi_sonuc = sonuclar_df.loc[en_iyi_idx]

print("🏆 EN İYİ GENEL YAPILANDIRMA")
print("="*50)
print(f"Model: {en_iyi_sonuc['Model']}")
print(f"Yöntem: {en_iyi_sonuc['Yöntem']}")
print(f"En İyi Skor: {en_iyi_sonuc['En İyi Skor']:.4f}")
print(f"Süre: {en_iyi_sonuc['Süre (s)']:.2f} saniye")

# Sonuçları kaydet
sonuclar_df.to_csv('../figures/optimizasyon_sonuclari.csv', index=False)
print("\\n✅ Sonuçlar '../figures/optimizasyon_sonuclari.csv' dosyasına kaydedildi")"""),

        markdown_cell("""---
## 9. Özet ve Sonuçlar

### Temel Bulgular

| Yöntem | Özellikler |
|--------|------------|
| **Grid Search** | Kapsamlı, yavaş ama eksiksiz |
| **Random Search** | Hızlı, iyi kapsam |
| **Bayesian Optimizasyon** | Akıllı, pahalı değerlendirmeler için verimli |

### Öneriler

| Senaryo | Önerilen Yöntem |
|---------|-----------------|
| Küçük parametre alanı | Grid Search |
| Büyük parametre alanı | Random Search veya Bayesian |
| Sınırlı hesaplama bütçesi | Bayesian Optimizasyon |
| Garantili kapsam gerekli | Grid Search |"""),

        code_cell("""print("="*60)
print("🎉 PROJE 1: OPTİMİZASYON - TAMAMLANDI!")
print("="*60)
print("\\n📁 '../figures/' klasörüne kaydedilen çıktı dosyaları:")
print("  - optimizasyon_karsilastirma_SVM.png")
print("  - optimizasyon_karsilastirma_RandomForest.png")
print("  - optimizasyon_karsilastirma_NeuralNetwork.png")
print("  - optimizasyon_sonuclari.csv")
print("\\n✅ Tüm deneyler başarıyla tamamlandı!")""")
    ]
    return create_notebook(cells)


# Save all notebooks
def main():
    # Project 1 - English
    print("Creating Project 1 notebooks...")
    
    nb1_en = create_project1_en()
    with open('Project_1_Optimization/notebooks/01_Optimization_EN.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb1_en, f, indent=2, ensure_ascii=False)
    print("  ✅ 01_Optimization_EN.ipynb created")
    
    nb1_tr = create_project1_tr()
    with open('Project_1_Optimization/notebooks/01_Optimization_TR.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb1_tr, f, indent=2, ensure_ascii=False)
    print("  ✅ 01_Optimization_TR.ipynb created")
    
    print("\n✅ Project 1 notebooks created successfully!")

if __name__ == "__main__":
    main()

