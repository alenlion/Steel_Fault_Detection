"""
Script to create notebooks for Project 2: Machine Learning
Creates both English and Turkish versions.
"""

import json

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def markdown_cell(content):
    return {"cell_type": "markdown", "metadata": {}, "source": content.split('\n')}

def code_cell(content):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": content.split('\n')}

# ============================================================
# PROJECT 2: MACHINE LEARNING - ENGLISH
# ============================================================
def create_project2_en():
    cells = [
        markdown_cell("""# Project 2: Machine Learning & Pattern Recognition

**Course**: Machine Learning

**Dataset**: Steel Plates Fault Detection

**Objective**: Multi-class classification using various ML algorithms

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Setup and Data Loading](#2-setup-and-data-loading)
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Training](#6-model-training)
7. [Model Comparison](#7-model-comparison)
8. [Feature Importance Analysis](#8-feature-importance-analysis)
9. [Summary and Conclusions](#9-summary-and-conclusions)"""),

        markdown_cell("""---
## 1. Introduction

### Problem Statement
Predict the type of fault in steel plates using machine learning classification algorithms.

### Algorithms Implemented:
1. **Logistic Regression** - Linear baseline
2. **K-Nearest Neighbors (KNN)** - Instance-based learning
3. **Decision Tree** - Rule-based classification
4. **Random Forest** - Ensemble of trees
5. **Gradient Boosting** - Sequential ensemble
6. **SVM** - Margin-based classifier
7. **Neural Network (MLP)** - Deep learning approach
8. **XGBoost** - Advanced boosting (if available)

### Evaluation Metrics:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (multi-class)
- Confusion Matrix"""),

        markdown_cell("""---
## 2. Setup and Data Loading"""),

        code_cell("""# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_auc_score)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Visualization settings
%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')
np.random.seed(42)

print("✅ Libraries imported successfully!")
print(f"XGBoost available: {XGBOOST_AVAILABLE}")"""),

        code_cell("""# Load data
df = pd.read_csv('../data/raw/steel_plates_fault.csv')
print(f"✅ Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
df.head()"""),

        markdown_cell("""---
## 3. Exploratory Data Analysis"""),

        code_cell("""# Dataset info
print("📊 Dataset Information:")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\\nMissing Values: {df.isnull().sum().sum()}")
print(f"\\nDuplicates: {df.duplicated().sum()}")
print(f"\\nData Types:")
print(df.dtypes.value_counts())"""),

        code_cell("""# Target distribution
print("🎯 Target Variable Distribution:")
class_counts = df['Class'].value_counts()
print(class_counts)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
colors = sns.color_palette('husl', len(class_counts))
bars = axes[0].bar(class_counts.index, class_counts.values, color=colors, edgecolor='black')
axes[0].set_xlabel('Fault Type', fontweight='bold')
axes[0].set_ylabel('Count', fontweight='bold')
axes[0].set_title('Class Distribution', fontweight='bold')
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Pie chart
axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', colors=colors)
axes[1].set_title('Class Distribution (%)', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        code_cell("""# Correlation heatmap
plt.figure(figsize=(16, 12))
corr = df.drop('Class', axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, annot=False,
            square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        code_cell("""# Statistical summary
print("📊 Statistical Summary:")
df.describe()"""),

        markdown_cell("""---
## 4. Data Preprocessing"""),

        code_cell("""# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

print(f"Features: {X.shape}")
print(f"Target: {y.shape}")
print(f"\\nFeature names: {X.columns.tolist()}")"""),

        code_cell("""# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("✅ Labels encoded!")
print(f"Classes: {encoder.classes_.tolist()}")
print(f"Encoded: {np.unique(y_encoded).tolist()}")"""),

        code_cell("""# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("✅ Data split completed!")
print(f"Training: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")"""),

        code_cell("""# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features scaled!")"""),

        markdown_cell("""---
## 5. Feature Engineering"""),

        code_cell("""# Create new features from existing ones
X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# Feature: Area ratio
if 'X_Maximum' in X.columns and 'X_Minimum' in X.columns:
    X_train_df['X_Range'] = X_train_df['X_Maximum'] - X_train_df['X_Minimum']
    X_test_df['X_Range'] = X_test_df['X_Maximum'] - X_test_df['X_Minimum']

if 'Y_Maximum' in X.columns and 'Y_Minimum' in X.columns:
    X_train_df['Y_Range'] = X_train_df['Y_Maximum'] - X_train_df['Y_Minimum']
    X_test_df['Y_Range'] = X_test_df['Y_Maximum'] - X_test_df['Y_Minimum']

# Feature: Luminosity range
if 'Maximum_of_Luminosity' in X.columns and 'Minimum_of_Luminosity' in X.columns:
    X_train_df['Luminosity_Range'] = X_train_df['Maximum_of_Luminosity'] - X_train_df['Minimum_of_Luminosity']
    X_test_df['Luminosity_Range'] = X_test_df['Maximum_of_Luminosity'] - X_test_df['Minimum_of_Luminosity']

# Feature: Aspect ratio
if 'X_Range' in X_train_df.columns and 'Y_Range' in X_train_df.columns:
    X_train_df['Aspect_Ratio'] = X_train_df['X_Range'] / (X_train_df['Y_Range'] + 0.001)
    X_test_df['Aspect_Ratio'] = X_test_df['X_Range'] / (X_test_df['Y_Range'] + 0.001)

print(f"✅ New features created!")
print(f"Original features: {X.shape[1]}")
print(f"Total features now: {X_train_df.shape[1]}")
print(f"\\nNew features: {[c for c in X_train_df.columns if c not in X.columns]}")"""),

        code_cell("""# Update training data
X_train_engineered = X_train_df.values
X_test_engineered = X_test_df.values

print(f"Final training shape: {X_train_engineered.shape}")
print(f"Final test shape: {X_test_engineered.shape}")"""),

        markdown_cell("""---
## 6. Model Training"""),

        code_cell("""# Define models
def get_models():
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'KNN (k=10)': KNeighborsClassifier(n_neighbors=10),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    
    return models

models = get_models()
print(f"📋 {len(models)} models to train:")
for name in models.keys():
    print(f"  - {name}")"""),

        code_cell("""# Train and evaluate all models
print("=" * 70)
print("MODEL TRAINING AND EVALUATION")
print("=" * 70)

results = []
trained_models = {}

for name, model in models.items():
    print(f"\\n🔄 Training {name}...")
    
    # Train
    model.fit(X_train_engineered, y_train)
    
    # Predict
    y_pred = model.predict(X_test_engineered)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # ROC-AUC
    roc_auc = None
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test_engineered)
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        except:
            pass
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })
    
    trained_models[name] = model
    print(f"  ✅ Accuracy: {acc:.4f}, F1: {f1:.4f}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
print("\\n✅ All models trained!")"""),

        markdown_cell("""---
## 7. Model Comparison"""),

        code_cell("""# Display results
print("📊 MODEL COMPARISON RESULTS")
print("=" * 80)
print(results_df.to_string(index=False))
print("=" * 80)

# Save results
results_df.to_csv('../figures/model_comparison_results.csv', index=False)
print("\\n✅ Results saved to '../figures/model_comparison_results.csv'")"""),

        code_cell("""# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy comparison
sorted_df = results_df.sort_values('Accuracy', ascending=True)
colors = sns.color_palette('viridis', len(sorted_df))

axes[0].barh(sorted_df['Model'], sorted_df['Accuracy'], color=colors, edgecolor='black')
axes[0].set_xlabel('Accuracy', fontweight='bold')
axes[0].set_title('Model Accuracy Comparison', fontweight='bold')
axes[0].set_xlim([0.5, 1.0])
for i, v in enumerate(sorted_df['Accuracy']):
    axes[0].text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')

# F1-Score comparison
sorted_f1 = results_df.sort_values('F1-Score', ascending=True)
axes[1].barh(sorted_f1['Model'], sorted_f1['F1-Score'], color=colors, edgecolor='black')
axes[1].set_xlabel('F1-Score', fontweight='bold')
axes[1].set_title('Model F1-Score Comparison', fontweight='bold')
axes[1].set_xlim([0.5, 1.0])
for i, v in enumerate(sorted_f1['F1-Score']):
    axes[1].text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/model_comparison_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        code_cell("""# Best model confusion matrix
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]

print(f"🏆 Best Model: {best_model_name}")
print(f"   Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")

# Confusion matrix
y_pred_best = best_model.predict(X_test_engineered)
cm = confusion_matrix(y_test, y_pred_best)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
axes[0].set_xlabel('Predicted', fontweight='bold')
axes[0].set_ylabel('Actual', fontweight='bold')
axes[0].set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold')

# Normalized
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
axes[1].set_xlabel('Predicted', fontweight='bold')
axes[1].set_ylabel('Actual', fontweight='bold')
axes[1].set_title(f'Normalized Confusion Matrix - {best_model_name}', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        code_cell("""# Classification report for best model
print(f"📊 Classification Report for {best_model_name}:")
print("=" * 70)
print(classification_report(y_test, y_pred_best, target_names=encoder.classes_))

# Save report
with open('../figures/classification_report.txt', 'w') as f:
    f.write(f"Best Model: {best_model_name}\\n")
    f.write("=" * 60 + "\\n")
    f.write(classification_report(y_test, y_pred_best, target_names=encoder.classes_))
print("\\n✅ Report saved to '../figures/classification_report.txt'")"""),

        markdown_cell("""---
## 8. Feature Importance Analysis"""),

        code_cell("""# Feature importance from Random Forest
rf_model = trained_models.get('Random Forest')

if rf_model:
    feature_names = list(X.columns) + [c for c in X_train_df.columns if c not in X.columns]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("📊 Top 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))
    
    # Save
    importance_df.to_csv('../figures/feature_importance.csv', index=False)
    
    # Visualize
    plt.figure(figsize=(12, 8))
    top_15 = importance_df.head(15)
    plt.barh(range(len(top_15)), top_15['Importance'], color='steelblue', edgecolor='black')
    plt.yticks(range(len(top_15)), top_15['Feature'])
    plt.gca().invert_yaxis()
    plt.xlabel('Importance', fontweight='bold')
    plt.title('Top 15 Feature Importances (Random Forest)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()"""),

        markdown_cell("""---
## 9. Summary and Conclusions

### Key Findings

1. **Best Performing Model**: Random Forest / Gradient Boosting typically perform best
2. **Feature Engineering**: Added features like X_Range, Y_Range, Aspect_Ratio improve performance
3. **Class Imbalance**: Some fault types are more common than others

### Model Comparison Summary

| Model Type | Characteristics |
|------------|-----------------|
| Logistic Regression | Fast, interpretable, linear baseline |
| KNN | Simple, sensitive to k value |
| Decision Tree | Interpretable, prone to overfitting |
| Random Forest | Robust, good generalization |
| Gradient Boosting | High accuracy, slower training |
| SVM | Good with clear margins |
| Neural Network | Flexible, needs more data |
| XGBoost | State-of-the-art for tabular data |"""),

        code_cell("""print("=" * 60)
print("🎉 PROJECT 2: MACHINE LEARNING - COMPLETE!")
print("=" * 60)
print(f"\\n🏆 Best Model: {best_model_name}")
print(f"   Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
print(f"   F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")
print("\\n📁 Output files in '../figures/':")
print("  - model_comparison_results.csv")
print("  - model_comparison_accuracy.png")
print("  - confusion_matrix.png")
print("  - classification_report.txt")
print("  - feature_importance.csv")
print("  - feature_importance.png")
print("\\n✅ All experiments completed successfully!")""")
    ]
    return create_notebook(cells)


# ============================================================
# PROJECT 2: MACHINE LEARNING - TURKISH
# ============================================================
def create_project2_tr():
    cells = [
        markdown_cell("""# Proje 2: Makine Öğrenimi ve Örüntü Tanıma

**Ders**: Makine Öğrenimi

**Veri Seti**: Çelik Levha Hata Tespiti

**Amaç**: Çeşitli ML algoritmaları ile çok sınıflı sınıflandırma

---

## İçindekiler

1. [Giriş](#1-giriş)
2. [Kurulum ve Veri Yükleme](#2-kurulum-ve-veri-yükleme)
3. [Keşifsel Veri Analizi](#3-keşifsel-veri-analizi)
4. [Veri Ön İşleme](#4-veri-ön-işleme)
5. [Özellik Mühendisliği](#5-özellik-mühendisliği)
6. [Model Eğitimi](#6-model-eğitimi)
7. [Model Karşılaştırması](#7-model-karşılaştırması)
8. [Özellik Önem Analizi](#8-özellik-önem-analizi)
9. [Özet ve Sonuçlar](#9-özet-ve-sonuçlar)"""),

        markdown_cell("""---
## 1. Giriş

### Problem Tanımı
Makine öğrenimi sınıflandırma algoritmaları kullanarak çelik levhalardaki hata türünü tahmin etme.

### Uygulanan Algoritmalar:
1. **Lojistik Regresyon** - Doğrusal temel
2. **K-En Yakın Komşu (KNN)** - Örnek tabanlı öğrenme
3. **Karar Ağacı** - Kural tabanlı sınıflandırma
4. **Random Forest** - Ağaç topluluğu
5. **Gradient Boosting** - Sıralı topluluk
6. **SVM** - Marj tabanlı sınıflandırıcı
7. **Yapay Sinir Ağı (MLP)** - Derin öğrenme yaklaşımı
8. **XGBoost** - Gelişmiş boosting

### Değerlendirme Metrikleri:
- Doğruluk, Kesinlik, Duyarlılık, F1-Skoru
- ROC-AUC (çok sınıflı)
- Karışıklık Matrisi"""),

        markdown_cell("""---
## 2. Kurulum ve Veri Yükleme"""),

        code_cell("""# Kütüphaneleri içe aktar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_auc_score)

# Modeller
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Görselleştirme ayarları
%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')
np.random.seed(42)

print("✅ Kütüphaneler başarıyla içe aktarıldı!")
print(f"XGBoost mevcut: {XGBOOST_AVAILABLE}")"""),

        code_cell("""# Veriyi yükle
df = pd.read_csv('../data/raw/steel_plates_fault.csv')
print(f"✅ Veri yüklendi: {df.shape[0]} örnek, {df.shape[1]} özellik")
df.head()"""),

        markdown_cell("""---
## 3. Keşifsel Veri Analizi"""),

        code_cell("""# Veri seti bilgileri
print("📊 Veri Seti Bilgileri:")
print("=" * 50)
print(f"Boyut: {df.shape}")
print(f"\\nEksik Değerler: {df.isnull().sum().sum()}")
print(f"\\nTekrarlanan: {df.duplicated().sum()}")"""),

        code_cell("""# Hedef dağılımı
print("🎯 Hedef Değişken Dağılımı:")
sinif_sayilari = df['Class'].value_counts()
print(sinif_sayilari)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Çubuk grafik
renkler = sns.color_palette('husl', len(sinif_sayilari))
cubuklar = axes[0].bar(sinif_sayilari.index, sinif_sayilari.values, color=renkler, edgecolor='black')
axes[0].set_xlabel('Hata Türü', fontweight='bold')
axes[0].set_ylabel('Sayı', fontweight='bold')
axes[0].set_title('Sınıf Dağılımı', fontweight='bold')
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Pasta grafik
axes[1].pie(sinif_sayilari.values, labels=sinif_sayilari.index, autopct='%1.1f%%', colors=renkler)
axes[1].set_title('Sınıf Dağılımı (%)', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/sinif_dagilimi.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        code_cell("""# Korelasyon ısı haritası
plt.figure(figsize=(16, 12))
korelasyon = df.drop('Class', axis=1).corr()
maske = np.triu(np.ones_like(korelasyon, dtype=bool))
sns.heatmap(korelasyon, mask=maske, cmap='coolwarm', center=0, annot=False,
            square=True, linewidths=0.5)
plt.title('Özellik Korelasyon Isı Haritası', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/korelasyon_haritasi.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""---
## 4. Veri Ön İşleme"""),

        code_cell("""# Özellikleri ve hedefi ayır
X = df.drop('Class', axis=1)
y = df['Class']

print(f"Özellikler: {X.shape}")
print(f"Hedef: {y.shape}")"""),

        code_cell("""# Etiketleri kodla
kodlayici = LabelEncoder()
y_kodlanmis = kodlayici.fit_transform(y)

print("✅ Etiketler kodlandı!")
print(f"Sınıflar: {kodlayici.classes_.tolist()}")"""),

        code_cell("""# Eğitim-test bölümü
X_egitim, X_test, y_egitim, y_test = train_test_split(
    X, y_kodlanmis, test_size=0.2, random_state=42, stratify=y_kodlanmis
)

print("✅ Veri bölme tamamlandı!")
print(f"Eğitim: {X_egitim.shape[0]} örnek")
print(f"Test: {X_test.shape[0]} örnek")"""),

        code_cell("""# Özellikleri ölçeklendir
olcekleyici = StandardScaler()
X_egitim_olcekli = olcekleyici.fit_transform(X_egitim)
X_test_olcekli = olcekleyici.transform(X_test)

print("✅ Özellikler ölçeklendirildi!")"""),

        markdown_cell("""---
## 5. Özellik Mühendisliği"""),

        code_cell("""# Mevcut özelliklerden yeni özellikler oluştur
X_egitim_df = pd.DataFrame(X_egitim_olcekli, columns=X.columns)
X_test_df = pd.DataFrame(X_test_olcekli, columns=X.columns)

# Özellik: Alan oranı
if 'X_Maximum' in X.columns and 'X_Minimum' in X.columns:
    X_egitim_df['X_Aralik'] = X_egitim_df['X_Maximum'] - X_egitim_df['X_Minimum']
    X_test_df['X_Aralik'] = X_test_df['X_Maximum'] - X_test_df['X_Minimum']

if 'Y_Maximum' in X.columns and 'Y_Minimum' in X.columns:
    X_egitim_df['Y_Aralik'] = X_egitim_df['Y_Maximum'] - X_egitim_df['Y_Minimum']
    X_test_df['Y_Aralik'] = X_test_df['Y_Maximum'] - X_test_df['Y_Minimum']

# Özellik: Parlaklık aralığı
if 'Maximum_of_Luminosity' in X.columns and 'Minimum_of_Luminosity' in X.columns:
    X_egitim_df['Parlaklik_Aralik'] = X_egitim_df['Maximum_of_Luminosity'] - X_egitim_df['Minimum_of_Luminosity']
    X_test_df['Parlaklik_Aralik'] = X_test_df['Maximum_of_Luminosity'] - X_test_df['Minimum_of_Luminosity']

# Özellik: En-boy oranı
if 'X_Aralik' in X_egitim_df.columns and 'Y_Aralik' in X_egitim_df.columns:
    X_egitim_df['EnBoy_Orani'] = X_egitim_df['X_Aralik'] / (X_egitim_df['Y_Aralik'] + 0.001)
    X_test_df['EnBoy_Orani'] = X_test_df['X_Aralik'] / (X_test_df['Y_Aralik'] + 0.001)

print(f"✅ Yeni özellikler oluşturuldu!")
print(f"Orijinal özellik sayısı: {X.shape[1]}")
print(f"Toplam özellik sayısı: {X_egitim_df.shape[1]}")"""),

        code_cell("""# Eğitim verilerini güncelle
X_egitim_muhendis = X_egitim_df.values
X_test_muhendis = X_test_df.values

print(f"Son eğitim boyutu: {X_egitim_muhendis.shape}")
print(f"Son test boyutu: {X_test_muhendis.shape}")"""),

        markdown_cell("""---
## 6. Model Eğitimi"""),

        code_cell("""# Modelleri tanımla
def modelleri_al():
    modeller = {
        'Lojistik Regresyon': LogisticRegression(max_iter=1000, random_state=42),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'KNN (k=10)': KNeighborsClassifier(n_neighbors=10),
        'Karar Ağacı': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'SVM (Doğrusal)': SVC(kernel='linear', probability=True, random_state=42),
        'Yapay Sinir Ağı': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    if XGBOOST_AVAILABLE:
        modeller['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    
    return modeller

modeller = modelleri_al()
print(f"📋 Eğitilecek {len(modeller)} model:")
for isim in modeller.keys():
    print(f"  - {isim}")"""),

        code_cell("""# Tüm modelleri eğit ve değerlendir
print("=" * 70)
print("MODEL EĞİTİMİ VE DEĞERLENDİRMESİ")
print("=" * 70)

sonuclar = []
egitilmis_modeller = {}

for isim, model in modeller.items():
    print(f"\\n🔄 {isim} eğitiliyor...")
    
    # Eğit
    model.fit(X_egitim_muhendis, y_egitim)
    
    # Tahmin
    y_tahmin = model.predict(X_test_muhendis)
    
    # Metrikler
    dogruluk = accuracy_score(y_test, y_tahmin)
    kesinlik = precision_score(y_test, y_tahmin, average='weighted', zero_division=0)
    duyarlilik = recall_score(y_test, y_tahmin, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_tahmin, average='weighted', zero_division=0)
    
    # ROC-AUC
    roc_auc = None
    if hasattr(model, 'predict_proba'):
        try:
            y_olasilik = model.predict_proba(X_test_muhendis)
            roc_auc = roc_auc_score(y_test, y_olasilik, multi_class='ovr', average='weighted')
        except:
            pass
    
    sonuclar.append({
        'Model': isim,
        'Doğruluk': dogruluk,
        'Kesinlik': kesinlik,
        'Duyarlılık': duyarlilik,
        'F1-Skor': f1,
        'ROC-AUC': roc_auc
    })
    
    egitilmis_modeller[isim] = model
    print(f"  ✅ Doğruluk: {dogruluk:.4f}, F1: {f1:.4f}")

sonuclar_df = pd.DataFrame(sonuclar)
sonuclar_df = sonuclar_df.sort_values('Doğruluk', ascending=False).reset_index(drop=True)
print("\\n✅ Tüm modeller eğitildi!")"""),

        markdown_cell("""---
## 7. Model Karşılaştırması"""),

        code_cell("""# Sonuçları göster
print("📊 MODEL KARŞILAŞTIRMA SONUÇLARI")
print("=" * 80)
print(sonuclar_df.to_string(index=False))
print("=" * 80)

# Sonuçları kaydet
sonuclar_df.to_csv('../figures/model_karsilastirma_sonuclari.csv', index=False)
print("\\n✅ Sonuçlar kaydedildi")"""),

        code_cell("""# Karşılaştırmayı görselleştir
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Doğruluk karşılaştırması
sirali_df = sonuclar_df.sort_values('Doğruluk', ascending=True)
renkler = sns.color_palette('viridis', len(sirali_df))

axes[0].barh(sirali_df['Model'], sirali_df['Doğruluk'], color=renkler, edgecolor='black')
axes[0].set_xlabel('Doğruluk', fontweight='bold')
axes[0].set_title('Model Doğruluk Karşılaştırması', fontweight='bold')
axes[0].set_xlim([0.5, 1.0])

# F1-Skor karşılaştırması
sirali_f1 = sonuclar_df.sort_values('F1-Skor', ascending=True)
axes[1].barh(sirali_f1['Model'], sirali_f1['F1-Skor'], color=renkler, edgecolor='black')
axes[1].set_xlabel('F1-Skor', fontweight='bold')
axes[1].set_title('Model F1-Skor Karşılaştırması', fontweight='bold')
axes[1].set_xlim([0.5, 1.0])

plt.tight_layout()
plt.savefig('../figures/model_karsilastirma.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        code_cell("""# En iyi model karışıklık matrisi
en_iyi_model_adi = sonuclar_df.iloc[0]['Model']
en_iyi_model = egitilmis_modeller[en_iyi_model_adi]

print(f"🏆 En İyi Model: {en_iyi_model_adi}")
print(f"   Doğruluk: {sonuclar_df.iloc[0]['Doğruluk']:.4f}")

# Karışıklık matrisi
y_tahmin_en_iyi = en_iyi_model.predict(X_test_muhendis)
km = confusion_matrix(y_test, y_tahmin_en_iyi)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Ham sayılar
sns.heatmap(km, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=kodlayici.classes_, yticklabels=kodlayici.classes_)
axes[0].set_xlabel('Tahmin', fontweight='bold')
axes[0].set_ylabel('Gerçek', fontweight='bold')
axes[0].set_title(f'Karışıklık Matrisi - {en_iyi_model_adi}', fontweight='bold')

# Normalize
km_norm = km.astype('float') / km.sum(axis=1)[:, np.newaxis]
sns.heatmap(km_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
            xticklabels=kodlayici.classes_, yticklabels=kodlayici.classes_)
axes[1].set_xlabel('Tahmin', fontweight='bold')
axes[1].set_ylabel('Gerçek', fontweight='bold')
axes[1].set_title(f'Normalize Karışıklık Matrisi - {en_iyi_model_adi}', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/karisiklik_matrisi.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""---
## 8. Özellik Önem Analizi"""),

        code_cell("""# Random Forest'tan özellik önemi
rf_model = egitilmis_modeller.get('Random Forest')

if rf_model:
    ozellik_adlari = list(X.columns) + [c for c in X_egitim_df.columns if c not in X.columns]
    
    onem_df = pd.DataFrame({
        'Özellik': ozellik_adlari,
        'Önem': rf_model.feature_importances_
    }).sort_values('Önem', ascending=False)
    
    print("📊 En Önemli 15 Özellik:")
    print(onem_df.head(15).to_string(index=False))
    
    # Kaydet
    onem_df.to_csv('../figures/ozellik_onemi.csv', index=False)
    
    # Görselleştir
    plt.figure(figsize=(12, 8))
    en_onemli_15 = onem_df.head(15)
    plt.barh(range(len(en_onemli_15)), en_onemli_15['Önem'], color='steelblue', edgecolor='black')
    plt.yticks(range(len(en_onemli_15)), en_onemli_15['Özellik'])
    plt.gca().invert_yaxis()
    plt.xlabel('Önem', fontweight='bold')
    plt.title('En Önemli 15 Özellik (Random Forest)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/ozellik_onemi.png', dpi=300, bbox_inches='tight')
    plt.show()"""),

        markdown_cell("""---
## 9. Özet ve Sonuçlar

### Temel Bulgular

1. **En İyi Performans Gösteren Model**: Random Forest / Gradient Boosting genellikle en iyi performansı gösterir
2. **Özellik Mühendisliği**: X_Aralik, Y_Aralik, EnBoy_Orani gibi eklenen özellikler performansı artırır
3. **Sınıf Dengesizliği**: Bazı hata türleri diğerlerinden daha yaygın

### Model Karşılaştırma Özeti

| Model Türü | Özellikler |
|------------|------------|
| Lojistik Regresyon | Hızlı, yorumlanabilir, doğrusal temel |
| KNN | Basit, k değerine duyarlı |
| Karar Ağacı | Yorumlanabilir, aşırı öğrenmeye eğilimli |
| Random Forest | Sağlam, iyi genelleme |
| Gradient Boosting | Yüksek doğruluk, yavaş eğitim |
| SVM | Net sınırlarla iyi çalışır |
| Yapay Sinir Ağı | Esnek, daha fazla veri gerektirir |
| XGBoost | Tablo verileri için en gelişmiş |"""),

        code_cell("""print("=" * 60)
print("🎉 PROJE 2: MAKİNE ÖĞRENİMİ - TAMAMLANDI!")
print("=" * 60)
print(f"\\n🏆 En İyi Model: {en_iyi_model_adi}")
print(f"   Doğruluk: {sonuclar_df.iloc[0]['Doğruluk']:.4f}")
print(f"   F1-Skor: {sonuclar_df.iloc[0]['F1-Skor']:.4f}")
print("\\n📁 '../figures/' klasöründeki çıktı dosyaları:")
print("  - model_karsilastirma_sonuclari.csv")
print("  - model_karsilastirma.png")
print("  - karisiklik_matrisi.png")
print("  - ozellik_onemi.csv")
print("  - ozellik_onemi.png")
print("\\n✅ Tüm deneyler başarıyla tamamlandı!")""")
    ]
    return create_notebook(cells)


def main():
    print("Creating Project 2 notebooks...")
    
    nb_en = create_project2_en()
    with open('Project_2_MachineLearning/notebooks/01_MachineLearning_EN.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb_en, f, indent=2, ensure_ascii=False)
    print("  ✅ 01_MachineLearning_EN.ipynb created")
    
    nb_tr = create_project2_tr()
    with open('Project_2_MachineLearning/notebooks/01_MachineLearning_TR.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb_tr, f, indent=2, ensure_ascii=False)
    print("  ✅ 01_MachineLearning_TR.ipynb created")
    
    print("\n✅ Project 2 notebooks created successfully!")

if __name__ == "__main__":
    main()

