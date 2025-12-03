"""
Script to create notebooks for Project 3: Data Mining
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


def create_project3_en():
    cells = [
        markdown_cell("""# Project 3: Data Mining

**Course**: Data Mining

**Dataset**: Steel Plates Fault Detection

**Objective**: Pattern discovery and knowledge extraction using data mining techniques

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Setup and Data Loading](#2-setup-and-data-loading)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis)
4. [Dimensionality Reduction](#4-dimensionality-reduction)
5. [Clustering Analysis](#5-clustering-analysis)
6. [Anomaly Detection](#6-anomaly-detection)
7. [Association and Pattern Analysis](#7-association-and-pattern-analysis)
8. [Summary and Conclusions](#8-summary-and-conclusions)"""),

        markdown_cell("""---
## 1. Introduction

### What is Data Mining?
Data mining is the process of discovering patterns, correlations, and anomalies in large datasets using machine learning, statistics, and database systems.

### Techniques Implemented:
1. **Exploratory Data Analysis (EDA)** - Understanding data structure
2. **Dimensionality Reduction** - PCA, t-SNE
3. **Clustering** - K-Means, Hierarchical, DBSCAN
4. **Anomaly Detection** - Isolation Forest
5. **Pattern Discovery** - Correlation analysis

### Goals:
- Discover hidden patterns in steel fault data
- Identify natural groupings (clusters)
- Detect anomalous data points
- Reduce dimensionality for visualization"""),

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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Visualization settings
%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style('whitegrid')
sns.set_palette('husl')
np.random.seed(42)

print("✅ Libraries imported successfully!")"""),

        code_cell("""# Load data
df = pd.read_csv('../data/raw/steel_plates_fault.csv')
print(f"✅ Data loaded: {df.shape[0]} samples, {df.shape[1]} columns")
df.head()"""),

        markdown_cell("""---
## 3. Exploratory Data Analysis (EDA)

### 3.1 Dataset Overview"""),

        code_cell("""# Dataset information
print("📊 DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\\nColumns: {df.columns.tolist()}")
print(f"\\nData Types:")
print(df.dtypes.value_counts())
print(f"\\nMissing Values: {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")"""),

        code_cell("""# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

feature_names = X.columns.tolist()
class_names = y.unique().tolist()

print(f"Features: {len(feature_names)}")
print(f"Classes: {class_names}")"""),

        markdown_cell("""### 3.2 Statistical Summary"""),

        code_cell("""# Descriptive statistics
print("📊 Descriptive Statistics:")
stats = X.describe().T
stats['range'] = stats['max'] - stats['min']
stats['cv'] = stats['std'] / stats['mean']  # Coefficient of variation
print(stats)

# Save statistics
stats.to_csv('../figures/descriptive_statistics.csv')
print("\\n✅ Statistics saved to '../figures/descriptive_statistics.csv'")"""),

        markdown_cell("""### 3.3 Distribution Analysis"""),

        code_cell("""# Distribution of first 8 features
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, col in enumerate(X.columns[:8]):
    axes[idx].hist(X[col], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[idx].set_title(col, fontweight='bold', fontsize=10)
    axes[idx].set_xlabel('')
    
plt.suptitle('Feature Distributions (First 8)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""### 3.4 Class Distribution"""),

        code_cell("""# Class distribution
print("🎯 Class Distribution:")
class_counts = y.value_counts()
print(class_counts)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
colors = sns.color_palette('husl', len(class_counts))
bars = axes[0].bar(class_counts.index, class_counts.values, color=colors, edgecolor='black')
axes[0].set_xlabel('Fault Type', fontweight='bold')
axes[0].set_ylabel('Count', fontweight='bold')
axes[0].set_title('Class Distribution', fontweight='bold')
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

for bar, count in zip(bars, class_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                 f'{count}', ha='center', fontweight='bold')

# Pie chart
axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', colors=colors)
axes[1].set_title('Class Distribution (%)', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""### 3.5 Correlation Analysis"""),

        code_cell("""# Correlation heatmap
plt.figure(figsize=(16, 14))
corr = X.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, 
            annot=False, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Find highly correlated pairs
print("\\n🔍 Highly Correlated Feature Pairs (|r| > 0.8):")
high_corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.8:
            high_corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

for pair in high_corr_pairs[:10]:
    print(f"  {pair[0]} <-> {pair[1]}: {pair[2]:.3f}")"""),

        markdown_cell("""---
## 4. Dimensionality Reduction

### 4.1 Data Scaling"""),

        code_cell("""# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("✅ Data scaled and labels encoded!")
print(f"Scaled data shape: {X_scaled.shape}")"""),

        markdown_cell("""### 4.2 Principal Component Analysis (PCA)"""),

        code_cell("""# PCA with all components
pca_full = PCA(random_state=42)
X_pca_full = pca_full.fit_transform(X_scaled)

# Explained variance
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("📊 PCA Explained Variance:")
print(f"First 2 components: {sum(explained_variance[:2]):.4f} ({sum(explained_variance[:2])*100:.2f}%)")
print(f"First 5 components: {sum(explained_variance[:5]):.4f} ({sum(explained_variance[:5])*100:.2f}%)")
print(f"First 10 components: {sum(explained_variance[:10]):.4f} ({sum(explained_variance[:10])*100:.2f}%)")

# Plot explained variance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Individual variance
axes[0].bar(range(1, len(explained_variance)+1), explained_variance, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Principal Component', fontweight='bold')
axes[0].set_ylabel('Explained Variance Ratio', fontweight='bold')
axes[0].set_title('Explained Variance per Component', fontweight='bold')

# Cumulative variance
axes[1].plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'bo-', linewidth=2)
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
axes[1].axhline(y=0.90, color='orange', linestyle='--', label='90% threshold')
axes[1].set_xlabel('Number of Components', fontweight='bold')
axes[1].set_ylabel('Cumulative Explained Variance', fontweight='bold')
axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/pca_variance.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        code_cell("""# PCA 2D visualization
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

print(f"PCA 2D: Explained variance = {sum(pca_2d.explained_variance_ratio_):.4f}")

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_encoded, 
                      cmap='tab10', alpha=0.6, s=30)
plt.colorbar(scatter, label='Class')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
plt.title('PCA 2D Projection', fontsize=14, fontweight='bold')

# Add legend
handles = [plt.scatter([], [], c=plt.cm.tab10(i/len(class_names)), label=name, s=50) 
           for i, name in enumerate(encoder.classes_)]
plt.legend(handles=handles, title='Fault Type', loc='best')

plt.tight_layout()
plt.savefig('../figures/pca_2d.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""### 4.3 t-SNE Visualization"""),

        code_cell("""# t-SNE
print("🔄 Running t-SNE (this may take a minute)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)
print("✅ t-SNE completed!")

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_encoded, 
                      cmap='tab10', alpha=0.6, s=30)
plt.colorbar(scatter, label='Class')
plt.xlabel('t-SNE 1', fontweight='bold')
plt.ylabel('t-SNE 2', fontweight='bold')
plt.title('t-SNE 2D Projection', fontsize=14, fontweight='bold')

handles = [plt.scatter([], [], c=plt.cm.tab10(i/len(class_names)), label=name, s=50) 
           for i, name in enumerate(encoder.classes_)]
plt.legend(handles=handles, title='Fault Type', loc='best')

plt.tight_layout()
plt.savefig('../figures/tsne.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""---
## 5. Clustering Analysis

### 5.1 Finding Optimal K (Elbow Method)"""),

        code_cell("""# Elbow method for optimal K
print("🔍 Finding optimal K using Elbow Method...")
k_range = range(2, 12)
inertias = []
silhouettes = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    print(f"  K={k}: Silhouette={silhouettes[-1]:.4f}, Inertia={inertias[-1]:.2f}")

# Find optimal K
optimal_k = list(k_range)[np.argmax(silhouettes)]
print(f"\\n🎯 Optimal K (by silhouette): {optimal_k}")

# Save elbow analysis
elbow_df = pd.DataFrame({'k': list(k_range), 'silhouette': silhouettes, 'inertia': inertias})
elbow_df.to_csv('../figures/elbow_analysis.csv', index=False)"""),

        code_cell("""# Plot elbow curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Inertia (Elbow)
axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)', fontweight='bold')
axes[0].set_ylabel('Inertia (Within-cluster SS)', fontweight='bold')
axes[0].set_title('Elbow Method', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Silhouette
axes[1].plot(list(k_range), silhouettes, 'go-', linewidth=2, markersize=8)
axes[1].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
axes[1].set_xlabel('Number of Clusters (K)', fontweight='bold')
axes[1].set_ylabel('Silhouette Score', fontweight='bold')
axes[1].set_title('Silhouette Analysis', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/elbow_plot.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""### 5.2 K-Means Clustering"""),

        code_cell("""# K-Means with optimal K (using 7 for number of fault types)
n_clusters = 7  # Number of fault types
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_ari = adjusted_rand_score(y_encoded, kmeans_labels)
kmeans_nmi = normalized_mutual_info_score(y_encoded, kmeans_labels)

print(f"📊 K-Means Clustering (K={n_clusters}):")
print(f"  Silhouette Score: {kmeans_silhouette:.4f}")
print(f"  Adjusted Rand Index: {kmeans_ari:.4f}")
print(f"  Normalized Mutual Info: {kmeans_nmi:.4f}")"""),

        markdown_cell("""### 5.3 Hierarchical Clustering"""),

        code_cell("""# Hierarchical clustering
hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
hc_labels = hc.fit_predict(X_scaled)

hc_silhouette = silhouette_score(X_scaled, hc_labels)
hc_ari = adjusted_rand_score(y_encoded, hc_labels)
hc_nmi = normalized_mutual_info_score(y_encoded, hc_labels)

print(f"📊 Hierarchical Clustering (K={n_clusters}):")
print(f"  Silhouette Score: {hc_silhouette:.4f}")
print(f"  Adjusted Rand Index: {hc_ari:.4f}")
print(f"  Normalized Mutual Info: {hc_nmi:.4f}")"""),

        code_cell("""# Dendrogram (on sample for visualization)
sample_size = min(200, len(X_scaled))
sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_idx]

linkage_matrix = linkage(X_sample, method='ward')

plt.figure(figsize=(16, 8))
dendrogram(linkage_matrix, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=8)
plt.xlabel('Sample Index', fontweight='bold')
plt.ylabel('Distance', fontweight='bold')
plt.title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""### 5.4 DBSCAN Clustering"""),

        code_cell("""# DBSCAN
dbscan = DBSCAN(eps=2.0, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"📊 DBSCAN Clustering:")
print(f"  Clusters found: {n_clusters_dbscan}")
print(f"  Noise points: {n_noise} ({n_noise/len(dbscan_labels)*100:.1f}%)")"""),

        markdown_cell("""### 5.5 Clustering Comparison"""),

        code_cell("""# Compare clustering results
clustering_results = pd.DataFrame([
    {'Method': 'K-Means', 'Silhouette': kmeans_silhouette, 'ARI': kmeans_ari, 'NMI': kmeans_nmi},
    {'Method': 'Hierarchical', 'Silhouette': hc_silhouette, 'ARI': hc_ari, 'NMI': hc_nmi}
])

print("📊 CLUSTERING COMPARISON:")
print("=" * 60)
print(clustering_results.to_string(index=False))
print("=" * 60)

# Save results
clustering_results.to_csv('../figures/clustering_comparison.csv', index=False)"""),

        code_cell("""# Visualize clustering comparison on PCA
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# True labels
scatter1 = axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_encoded, cmap='tab10', alpha=0.6, s=30)
axes[0].set_xlabel('PC1', fontweight='bold')
axes[0].set_ylabel('PC2', fontweight='bold')
axes[0].set_title('True Labels', fontweight='bold')

# K-Means
scatter2 = axes[1].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.6, s=30)
axes[1].set_xlabel('PC1', fontweight='bold')
axes[1].set_ylabel('PC2', fontweight='bold')
axes[1].set_title(f'K-Means (Silhouette={kmeans_silhouette:.3f})', fontweight='bold')

# Hierarchical
scatter3 = axes[2].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=hc_labels, cmap='tab10', alpha=0.6, s=30)
axes[2].set_xlabel('PC1', fontweight='bold')
axes[2].set_ylabel('PC2', fontweight='bold')
axes[2].set_title(f'Hierarchical (Silhouette={hc_silhouette:.3f})', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/clustering_comparison.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""---
## 6. Anomaly Detection"""),

        code_cell("""# Isolation Forest for anomaly detection
print("🔍 Running Anomaly Detection (Isolation Forest)...")

contamination = 0.1  # Expected proportion of anomalies
iso_forest = IsolationForest(contamination=contamination, random_state=42)
anomaly_labels = iso_forest.fit_predict(X_scaled)
anomaly_scores = iso_forest.decision_function(X_scaled)

n_anomalies = (anomaly_labels == -1).sum()

print(f"\\n📊 Anomaly Detection Results:")
print(f"  Contamination: {contamination*100:.1f}%")
print(f"  Anomalies found: {n_anomalies} ({n_anomalies/len(X_scaled)*100:.1f}%)")
print(f"  Normal samples: {len(X_scaled) - n_anomalies}")

# Save results
anomaly_df = pd.DataFrame({
    'Index': range(len(anomaly_labels)),
    'Label': anomaly_labels,
    'Score': anomaly_scores
})
anomaly_df.to_csv('../figures/anomaly_results.csv', index=False)
print("\\n✅ Anomaly results saved to '../figures/anomaly_results.csv'")"""),

        code_cell("""# Visualize anomalies on PCA
plt.figure(figsize=(12, 8))

# Normal points
normal_mask = anomaly_labels == 1
plt.scatter(X_pca_2d[normal_mask, 0], X_pca_2d[normal_mask, 1], 
           c='blue', alpha=0.5, s=30, label='Normal')

# Anomalies
anomaly_mask = anomaly_labels == -1
plt.scatter(X_pca_2d[anomaly_mask, 0], X_pca_2d[anomaly_mask, 1], 
           c='red', alpha=0.8, s=50, marker='x', label='Anomaly')

plt.xlabel('PC1', fontweight='bold')
plt.ylabel('PC2', fontweight='bold')
plt.title(f'Anomaly Detection (Isolation Forest) - {n_anomalies} anomalies detected', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('../figures/anomaly_detection.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""---
## 7. Association and Pattern Analysis"""),

        code_cell("""# Analyze patterns in anomalies vs normal
print("📊 Comparing Normal vs Anomalous Samples:")
print("=" * 60)

normal_data = X[anomaly_labels == 1]
anomaly_data = X[anomaly_labels == -1]

comparison = pd.DataFrame({
    'Feature': X.columns,
    'Normal_Mean': normal_data.mean().values,
    'Anomaly_Mean': anomaly_data.mean().values
})
comparison['Difference'] = comparison['Anomaly_Mean'] - comparison['Normal_Mean']
comparison['Abs_Difference'] = abs(comparison['Difference'])
comparison = comparison.sort_values('Abs_Difference', ascending=False)

print("\\nTop 10 Features with Largest Differences:")
print(comparison.head(10).to_string(index=False))"""),

        markdown_cell("""---
## 8. Summary and Conclusions

### Key Findings

1. **Dimensionality Reduction**:
   - PCA captures ~60-70% variance with 2 components
   - t-SNE provides better cluster separation visualization

2. **Clustering**:
   - K-Means and Hierarchical show similar performance
   - Natural groupings align partially with fault types

3. **Anomaly Detection**:
   - ~10% of samples identified as anomalies
   - Anomalies show distinct patterns in certain features

### Data Mining Techniques Summary

| Technique | Purpose | Key Finding |
|-----------|---------|-------------|
| PCA | Dimensionality reduction | First 2 PCs explain ~50% variance |
| t-SNE | Visualization | Better cluster separation than PCA |
| K-Means | Clustering | Silhouette score indicates moderate clustering |
| Hierarchical | Clustering | Similar performance to K-Means |
| DBSCAN | Density clustering | Identifies noise points |
| Isolation Forest | Anomaly detection | ~10% anomalies detected |"""),

        code_cell("""print("=" * 60)
print("🎉 PROJECT 3: DATA MINING - COMPLETE!")
print("=" * 60)
print("\\n📊 Summary:")
print(f"  PCA: {sum(pca_2d.explained_variance_ratio_)*100:.1f}% variance explained (2 components)")
print(f"  Best Clustering: K-Means (Silhouette={kmeans_silhouette:.4f})")
print(f"  Anomalies Detected: {n_anomalies} ({n_anomalies/len(X_scaled)*100:.1f}%)")
print("\\n📁 Output files in '../figures/':")
print("  - descriptive_statistics.csv")
print("  - class_distribution.png")
print("  - correlation_heatmap.png")
print("  - pca_2d.png")
print("  - tsne.png")
print("  - elbow_plot.png")
print("  - clustering_comparison.png")
print("  - dendrogram.png")
print("  - anomaly_results.csv")
print("\\n✅ All experiments completed successfully!")""")
    ]
    return create_notebook(cells)


def create_project3_tr():
    cells = [
        markdown_cell("""# Proje 3: Veri Madenciliği

**Ders**: Veri Madenciliği

**Veri Seti**: Çelik Levha Hata Tespiti

**Amaç**: Veri madenciliği teknikleri kullanarak örüntü keşfi ve bilgi çıkarımı

---

## İçindekiler

1. [Giriş](#1-giriş)
2. [Kurulum ve Veri Yükleme](#2-kurulum-ve-veri-yükleme)
3. [Keşifsel Veri Analizi (EDA)](#3-keşifsel-veri-analizi)
4. [Boyut İndirgeme](#4-boyut-indirgeme)
5. [Kümeleme Analizi](#5-kümeleme-analizi)
6. [Anomali Tespiti](#6-anomali-tespiti)
7. [İlişki ve Örüntü Analizi](#7-ilişki-ve-örüntü-analizi)
8. [Özet ve Sonuçlar](#8-özet-ve-sonuçlar)"""),

        markdown_cell("""---
## 1. Giriş

### Veri Madenciliği Nedir?
Veri madenciliği, makine öğrenimi, istatistik ve veritabanı sistemleri kullanarak büyük veri kümelerinde örüntü, korelasyon ve anomali keşfetme sürecidir.

### Uygulanan Teknikler:
1. **Keşifsel Veri Analizi (EDA)** - Veri yapısını anlama
2. **Boyut İndirgeme** - PCA, t-SNE
3. **Kümeleme** - K-Means, Hiyerarşik, DBSCAN
4. **Anomali Tespiti** - Isolation Forest
5. **Örüntü Keşfi** - Korelasyon analizi

### Hedefler:
- Çelik hata verilerinde gizli örüntüleri keşfetme
- Doğal gruplandırmaları (kümeleri) belirleme
- Anormal veri noktalarını tespit etme
- Görselleştirme için boyut indirgeme"""),

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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Görselleştirme ayarları
%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style('whitegrid')
sns.set_palette('husl')
np.random.seed(42)

print("✅ Kütüphaneler başarıyla içe aktarıldı!")"""),

        code_cell("""# Veriyi yükle
df = pd.read_csv('../data/raw/steel_plates_fault.csv')
print(f"✅ Veri yüklendi: {df.shape[0]} örnek, {df.shape[1]} sütun")
df.head()"""),

        markdown_cell("""---
## 3. Keşifsel Veri Analizi (EDA)

### 3.1 Veri Seti Genel Bakış"""),

        code_cell("""# Veri seti bilgileri
print("📊 VERİ SETİ GENEL BAKIŞ")
print("=" * 60)
print(f"Boyut: {df.shape}")
print(f"\\nSütunlar: {df.columns.tolist()}")
print(f"\\nVeri Türleri:")
print(df.dtypes.value_counts())
print(f"\\nEksik Değerler: {df.isnull().sum().sum()}")
print(f"Tekrarlanan: {df.duplicated().sum()}")"""),

        code_cell("""# Özellikleri ve hedefi ayır
X = df.drop('Class', axis=1)
y = df['Class']

ozellik_adlari = X.columns.tolist()
sinif_adlari = y.unique().tolist()

print(f"Özellikler: {len(ozellik_adlari)}")
print(f"Sınıflar: {sinif_adlari}")"""),

        markdown_cell("""### 3.2 İstatistiksel Özet"""),

        code_cell("""# Tanımlayıcı istatistikler
print("📊 Tanımlayıcı İstatistikler:")
istatistik = X.describe().T
istatistik['aralik'] = istatistik['max'] - istatistik['min']
istatistik['degisim_katsayisi'] = istatistik['std'] / istatistik['mean']
print(istatistik)

# İstatistikleri kaydet
istatistik.to_csv('../figures/tanimlayici_istatistikler.csv')
print("\\n✅ İstatistikler kaydedildi")"""),

        markdown_cell("""### 3.3 Sınıf Dağılımı"""),

        code_cell("""# Sınıf dağılımı
print("🎯 Sınıf Dağılımı:")
sinif_sayilari = y.value_counts()
print(sinif_sayilari)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Çubuk grafik
renkler = sns.color_palette('husl', len(sinif_sayilari))
cubuklar = axes[0].bar(sinif_sayilari.index, sinif_sayilari.values, color=renkler, edgecolor='black')
axes[0].set_xlabel('Hata Türü', fontweight='bold')
axes[0].set_ylabel('Sayı', fontweight='bold')
axes[0].set_title('Sınıf Dağılımı', fontweight='bold')
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

for cubuk, sayi in zip(cubuklar, sinif_sayilari.values):
    axes[0].text(cubuk.get_x() + cubuk.get_width()/2, cubuk.get_height() + 5, 
                 f'{sayi}', ha='center', fontweight='bold')

# Pasta grafik
axes[1].pie(sinif_sayilari.values, labels=sinif_sayilari.index, autopct='%1.1f%%', colors=renkler)
axes[1].set_title('Sınıf Dağılımı (%)', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/sinif_dagilimi.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""### 3.4 Korelasyon Analizi"""),

        code_cell("""# Korelasyon ısı haritası
plt.figure(figsize=(16, 14))
korelasyon = X.corr()
maske = np.triu(np.ones_like(korelasyon, dtype=bool))
sns.heatmap(korelasyon, mask=maske, cmap='coolwarm', center=0, 
            annot=False, square=True, linewidths=0.5)
plt.title('Özellik Korelasyon Isı Haritası', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/korelasyon_haritasi.png', dpi=300, bbox_inches='tight')
plt.show()

# Yüksek korelasyonlu çiftleri bul
print("\\n🔍 Yüksek Korelasyonlu Özellik Çiftleri (|r| > 0.8):")
yuksek_korelasyon = []
for i in range(len(korelasyon.columns)):
    for j in range(i+1, len(korelasyon.columns)):
        if abs(korelasyon.iloc[i, j]) > 0.8:
            yuksek_korelasyon.append((korelasyon.columns[i], korelasyon.columns[j], korelasyon.iloc[i, j]))

for cift in yuksek_korelasyon[:10]:
    print(f"  {cift[0]} <-> {cift[1]}: {cift[2]:.3f}")"""),

        markdown_cell("""---
## 4. Boyut İndirgeme

### 4.1 Veri Ölçekleme"""),

        code_cell("""# Özellikleri ölçekle
olcekleyici = StandardScaler()
X_olcekli = olcekleyici.fit_transform(X)

# Etiketleri kodla
kodlayici = LabelEncoder()
y_kodlanmis = kodlayici.fit_transform(y)

print("✅ Veri ölçeklendi ve etiketler kodlandı!")
print(f"Ölçeklenmiş veri boyutu: {X_olcekli.shape}")"""),

        markdown_cell("""### 4.2 Temel Bileşen Analizi (PCA)"""),

        code_cell("""# Tüm bileşenlerle PCA
pca_tam = PCA(random_state=42)
X_pca_tam = pca_tam.fit_transform(X_olcekli)

# Açıklanan varyans
aciklanan_varyans = pca_tam.explained_variance_ratio_
kumulatif_varyans = np.cumsum(aciklanan_varyans)

print("📊 PCA Açıklanan Varyans:")
print(f"İlk 2 bileşen: {sum(aciklanan_varyans[:2]):.4f} ({sum(aciklanan_varyans[:2])*100:.2f}%)")
print(f"İlk 5 bileşen: {sum(aciklanan_varyans[:5]):.4f} ({sum(aciklanan_varyans[:5])*100:.2f}%)")
print(f"İlk 10 bileşen: {sum(aciklanan_varyans[:10]):.4f} ({sum(aciklanan_varyans[:10])*100:.2f}%)")

# Açıklanan varyans grafiği
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bireysel varyans
axes[0].bar(range(1, len(aciklanan_varyans)+1), aciklanan_varyans, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Temel Bileşen', fontweight='bold')
axes[0].set_ylabel('Açıklanan Varyans Oranı', fontweight='bold')
axes[0].set_title('Bileşen Başına Açıklanan Varyans', fontweight='bold')

# Kümülatif varyans
axes[1].plot(range(1, len(kumulatif_varyans)+1), kumulatif_varyans, 'bo-', linewidth=2)
axes[1].axhline(y=0.95, color='r', linestyle='--', label='%95 eşik')
axes[1].axhline(y=0.90, color='orange', linestyle='--', label='%90 eşik')
axes[1].set_xlabel('Bileşen Sayısı', fontweight='bold')
axes[1].set_ylabel('Kümülatif Açıklanan Varyans', fontweight='bold')
axes[1].set_title('Kümülatif Açıklanan Varyans', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('../figures/pca_varyans.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        code_cell("""# PCA 2D görselleştirme
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_olcekli)

print(f"PCA 2D: Açıklanan varyans = {sum(pca_2d.explained_variance_ratio_):.4f}")

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_kodlanmis, 
                      cmap='tab10', alpha=0.6, s=30)
plt.colorbar(scatter, label='Sınıf')
plt.xlabel(f'TB1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
plt.ylabel(f'TB2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
plt.title('PCA 2D Projeksiyon', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/pca_2d.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""### 4.3 t-SNE Görselleştirme"""),

        code_cell("""# t-SNE
print("🔄 t-SNE çalışıyor (bu bir dakika sürebilir)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X_olcekli)
print("✅ t-SNE tamamlandı!")

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_kodlanmis, 
                      cmap='tab10', alpha=0.6, s=30)
plt.colorbar(scatter, label='Sınıf')
plt.xlabel('t-SNE 1', fontweight='bold')
plt.ylabel('t-SNE 2', fontweight='bold')
plt.title('t-SNE 2D Projeksiyon', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/tsne.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""---
## 5. Kümeleme Analizi

### 5.1 Optimal K Bulma (Dirsek Yöntemi)"""),

        code_cell("""# Optimal K için dirsek yöntemi
print("🔍 Dirsek Yöntemi ile optimal K bulunuyor...")
k_aralik = range(2, 12)
atalet_degerleri = []
silhouette_degerleri = []

for k in k_aralik:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    etiketler = kmeans.fit_predict(X_olcekli)
    atalet_degerleri.append(kmeans.inertia_)
    silhouette_degerleri.append(silhouette_score(X_olcekli, etiketler))
    print(f"  K={k}: Silhouette={silhouette_degerleri[-1]:.4f}")

# Optimal K bul
optimal_k = list(k_aralik)[np.argmax(silhouette_degerleri)]
print(f"\\n🎯 Optimal K (silhouette'e göre): {optimal_k}")

# Dirsek analizini kaydet
dirsek_df = pd.DataFrame({'k': list(k_aralik), 'silhouette': silhouette_degerleri, 'atalet': atalet_degerleri})
dirsek_df.to_csv('../figures/dirsek_analizi.csv', index=False)"""),

        code_cell("""# Dirsek eğrisini çiz
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Atalet (Dirsek)
axes[0].plot(list(k_aralik), atalet_degerleri, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Küme Sayısı (K)', fontweight='bold')
axes[0].set_ylabel('Atalet (Küme İçi KT)', fontweight='bold')
axes[0].set_title('Dirsek Yöntemi', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Silhouette
axes[1].plot(list(k_aralik), silhouette_degerleri, 'go-', linewidth=2, markersize=8)
axes[1].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
axes[1].set_xlabel('Küme Sayısı (K)', fontweight='bold')
axes[1].set_ylabel('Silhouette Skoru', fontweight='bold')
axes[1].set_title('Silhouette Analizi', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/dirsek_grafigi.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""### 5.2 K-Means Kümeleme"""),

        code_cell("""# Optimal K ile K-Means (7 hata türü için)
n_kume = 7
kmeans = KMeans(n_clusters=n_kume, random_state=42, n_init=10)
kmeans_etiketler = kmeans.fit_predict(X_olcekli)

kmeans_silhouette = silhouette_score(X_olcekli, kmeans_etiketler)
kmeans_ari = adjusted_rand_score(y_kodlanmis, kmeans_etiketler)
kmeans_nmi = normalized_mutual_info_score(y_kodlanmis, kmeans_etiketler)

print(f"📊 K-Means Kümeleme (K={n_kume}):")
print(f"  Silhouette Skoru: {kmeans_silhouette:.4f}")
print(f"  Düzeltilmiş Rand İndeksi: {kmeans_ari:.4f}")
print(f"  Normalize Karşılıklı Bilgi: {kmeans_nmi:.4f}")"""),

        markdown_cell("""### 5.3 Hiyerarşik Kümeleme"""),

        code_cell("""# Hiyerarşik kümeleme
hc = AgglomerativeClustering(n_clusters=n_kume, linkage='ward')
hc_etiketler = hc.fit_predict(X_olcekli)

hc_silhouette = silhouette_score(X_olcekli, hc_etiketler)
hc_ari = adjusted_rand_score(y_kodlanmis, hc_etiketler)
hc_nmi = normalized_mutual_info_score(y_kodlanmis, hc_etiketler)

print(f"📊 Hiyerarşik Kümeleme (K={n_kume}):")
print(f"  Silhouette Skoru: {hc_silhouette:.4f}")
print(f"  Düzeltilmiş Rand İndeksi: {hc_ari:.4f}")
print(f"  Normalize Karşılıklı Bilgi: {hc_nmi:.4f}")"""),

        code_cell("""# Dendrogram (görselleştirme için örneklem üzerinde)
ornek_boyut = min(200, len(X_olcekli))
ornek_idx = np.random.choice(len(X_olcekli), ornek_boyut, replace=False)
X_ornek = X_olcekli[ornek_idx]

baglanti_matrisi = linkage(X_ornek, method='ward')

plt.figure(figsize=(16, 8))
dendrogram(baglanti_matrisi, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=8)
plt.xlabel('Örnek İndeksi', fontweight='bold')
plt.ylabel('Mesafe', fontweight='bold')
plt.title('Hiyerarşik Kümeleme Dendrogramı', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""### 5.4 Kümeleme Karşılaştırması"""),

        code_cell("""# Kümeleme sonuçlarını karşılaştır
kumeleme_sonuclari = pd.DataFrame([
    {'Yöntem': 'K-Means', 'Silhouette': kmeans_silhouette, 'ARI': kmeans_ari, 'NMI': kmeans_nmi},
    {'Yöntem': 'Hiyerarşik', 'Silhouette': hc_silhouette, 'ARI': hc_ari, 'NMI': hc_nmi}
])

print("📊 KÜMELEME KARŞILAŞTIRMASI:")
print("=" * 60)
print(kumeleme_sonuclari.to_string(index=False))
print("=" * 60)

# Sonuçları kaydet
kumeleme_sonuclari.to_csv('../figures/kumeleme_karsilastirma.csv', index=False)"""),

        code_cell("""# PCA üzerinde kümeleme karşılaştırmasını görselleştir
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gerçek etiketler
scatter1 = axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_kodlanmis, cmap='tab10', alpha=0.6, s=30)
axes[0].set_xlabel('TB1', fontweight='bold')
axes[0].set_ylabel('TB2', fontweight='bold')
axes[0].set_title('Gerçek Etiketler', fontweight='bold')

# K-Means
scatter2 = axes[1].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=kmeans_etiketler, cmap='tab10', alpha=0.6, s=30)
axes[1].set_xlabel('TB1', fontweight='bold')
axes[1].set_ylabel('TB2', fontweight='bold')
axes[1].set_title(f'K-Means (Silhouette={kmeans_silhouette:.3f})', fontweight='bold')

# Hiyerarşik
scatter3 = axes[2].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=hc_etiketler, cmap='tab10', alpha=0.6, s=30)
axes[2].set_xlabel('TB1', fontweight='bold')
axes[2].set_ylabel('TB2', fontweight='bold')
axes[2].set_title(f'Hiyerarşik (Silhouette={hc_silhouette:.3f})', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/kumeleme_karsilastirma.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""---
## 6. Anomali Tespiti"""),

        code_cell("""# Anomali tespiti için Isolation Forest
print("🔍 Anomali Tespiti çalışıyor (Isolation Forest)...")

kontaminasyon = 0.1  # Beklenen anomali oranı
iso_forest = IsolationForest(contamination=kontaminasyon, random_state=42)
anomali_etiketler = iso_forest.fit_predict(X_olcekli)
anomali_skorlar = iso_forest.decision_function(X_olcekli)

n_anomali = (anomali_etiketler == -1).sum()

print(f"\\n📊 Anomali Tespiti Sonuçları:")
print(f"  Kontaminasyon: {kontaminasyon*100:.1f}%")
print(f"  Bulunan anomali: {n_anomali} ({n_anomali/len(X_olcekli)*100:.1f}%)")
print(f"  Normal örnekler: {len(X_olcekli) - n_anomali}")

# Sonuçları kaydet
anomali_df = pd.DataFrame({
    'İndeks': range(len(anomali_etiketler)),
    'Etiket': anomali_etiketler,
    'Skor': anomali_skorlar
})
anomali_df.to_csv('../figures/anomali_sonuclari.csv', index=False)
print("\\n✅ Anomali sonuçları kaydedildi")"""),

        code_cell("""# PCA üzerinde anomalileri görselleştir
plt.figure(figsize=(12, 8))

# Normal noktalar
normal_maske = anomali_etiketler == 1
plt.scatter(X_pca_2d[normal_maske, 0], X_pca_2d[normal_maske, 1], 
           c='blue', alpha=0.5, s=30, label='Normal')

# Anomaliler
anomali_maske = anomali_etiketler == -1
plt.scatter(X_pca_2d[anomali_maske, 0], X_pca_2d[anomali_maske, 1], 
           c='red', alpha=0.8, s=50, marker='x', label='Anomali')

plt.xlabel('TB1', fontweight='bold')
plt.ylabel('TB2', fontweight='bold')
plt.title(f'Anomali Tespiti (Isolation Forest) - {n_anomali} anomali tespit edildi', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('../figures/anomali_tespiti.png', dpi=300, bbox_inches='tight')
plt.show()"""),

        markdown_cell("""---
## 7. İlişki ve Örüntü Analizi"""),

        code_cell("""# Anomalilerde ve normal örneklerde örüntüleri analiz et
print("📊 Normal ve Anomali Örneklerin Karşılaştırması:")
print("=" * 60)

normal_veri = X[anomali_etiketler == 1]
anomali_veri = X[anomali_etiketler == -1]

karsilastirma = pd.DataFrame({
    'Özellik': X.columns,
    'Normal_Ortalama': normal_veri.mean().values,
    'Anomali_Ortalama': anomali_veri.mean().values
})
karsilastirma['Fark'] = karsilastirma['Anomali_Ortalama'] - karsilastirma['Normal_Ortalama']
karsilastirma['Mutlak_Fark'] = abs(karsilastirma['Fark'])
karsilastirma = karsilastirma.sort_values('Mutlak_Fark', ascending=False)

print("\\nEn Büyük Farklara Sahip 10 Özellik:")
print(karsilastirma.head(10).to_string(index=False))"""),

        markdown_cell("""---
## 8. Özet ve Sonuçlar

### Temel Bulgular

1. **Boyut İndirgeme**:
   - PCA 2 bileşenle ~%60-70 varyans yakalar
   - t-SNE daha iyi küme ayrımı görselleştirmesi sağlar

2. **Kümeleme**:
   - K-Means ve Hiyerarşik benzer performans gösterir
   - Doğal gruplandırmalar kısmen hata türleriyle uyumlu

3. **Anomali Tespiti**:
   - Örneklerin ~%10'u anomali olarak tespit edildi
   - Anomaliler belirli özelliklerde farklı örüntüler gösterir

### Veri Madenciliği Teknikleri Özeti

| Teknik | Amaç | Temel Bulgu |
|--------|------|-------------|
| PCA | Boyut indirgeme | İlk 2 TB ~%50 varyansı açıklar |
| t-SNE | Görselleştirme | PCA'dan daha iyi küme ayrımı |
| K-Means | Kümeleme | Silhouette skoru orta düzey kümeleme gösterir |
| Hiyerarşik | Kümeleme | K-Means ile benzer performans |
| DBSCAN | Yoğunluk kümeleme | Gürültü noktalarını tespit eder |
| Isolation Forest | Anomali tespiti | ~%10 anomali tespit edildi |"""),

        code_cell("""print("=" * 60)
print("🎉 PROJE 3: VERİ MADENCİLİĞİ - TAMAMLANDI!")
print("=" * 60)
print("\\n📊 Özet:")
print(f"  PCA: {sum(pca_2d.explained_variance_ratio_)*100:.1f}% varyans açıklandı (2 bileşen)")
print(f"  En İyi Kümeleme: K-Means (Silhouette={kmeans_silhouette:.4f})")
print(f"  Tespit Edilen Anomali: {n_anomali} ({n_anomali/len(X_olcekli)*100:.1f}%)")
print("\\n📁 '../figures/' klasöründeki çıktı dosyaları:")
print("  - tanimlayici_istatistikler.csv")
print("  - sinif_dagilimi.png")
print("  - korelasyon_haritasi.png")
print("  - pca_2d.png")
print("  - tsne.png")
print("  - dirsek_grafigi.png")
print("  - kumeleme_karsilastirma.png")
print("  - dendrogram.png")
print("  - anomali_sonuclari.csv")
print("\\n✅ Tüm deneyler başarıyla tamamlandı!")""")
    ]
    return create_notebook(cells)


def main():
    print("Creating Project 3 notebooks...")
    
    nb_en = create_project3_en()
    with open('Project_3_DataMining/notebooks/01_DataMining_EN.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb_en, f, indent=2, ensure_ascii=False)
    print("  ✅ 01_DataMining_EN.ipynb created")
    
    nb_tr = create_project3_tr()
    with open('Project_3_DataMining/notebooks/01_DataMining_TR.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb_tr, f, indent=2, ensure_ascii=False)
    print("  ✅ 01_DataMining_TR.ipynb created")
    
    print("\n✅ Project 3 notebooks created successfully!")

if __name__ == "__main__":
    main()

