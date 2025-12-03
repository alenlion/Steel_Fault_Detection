# ===========================================
# Visualization Module - Project 3: Data Mining
# ===========================================

"""
Module for data mining visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram


def plot_correlation_heatmap(X, feature_names=None, save_path=None):
    """
    Plot correlation heatmap.
    """
    
    if isinstance(X, np.ndarray):
        if feature_names:
            df = pd.DataFrame(X, columns=feature_names)
        else:
            df = pd.DataFrame(X)
    else:
        df = X
    
    corr = df.corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, ax=ax)
    
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_class_distribution(y, class_names=None, save_path=None):
    """
    Plot class distribution.
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if isinstance(y, pd.Series):
        counts = y.value_counts().sort_index()
    else:
        unique, counts_arr = np.unique(y, return_counts=True)
        counts = pd.Series(counts_arr, index=unique)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
    bars = ax.bar(range(len(counts)), counts.values, color=colors)
    
    if class_names:
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
    
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', fontweight='bold')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_pca_2d(X_pca, y, class_names=None, save_path=None):
    """
    Plot 2D PCA visualization.
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    classes = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))
    
    for cls, color in zip(classes, colors):
        mask = y == cls
        label = class_names[cls] if class_names else str(cls)
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], label=label, alpha=0.6, s=50)
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('PCA Visualization (2D)', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_tsne(X_tsne, y, class_names=None, save_path=None):
    """
    Plot t-SNE visualization.
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    classes = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))
    
    for cls, color in zip(classes, colors):
        mask = y == cls
        label = class_names[cls] if class_names else str(cls)
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[color], label=label, alpha=0.6, s=50)
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('t-SNE Visualization', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_elbow(results_df, save_path=None):
    """
    Plot elbow curve and silhouette scores.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(results_df['k'], results_df['inertia'], 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Inertia', fontsize=12)
    axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(results_df['k'], results_df['silhouette'], 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_dendrogram(linkage_matrix, save_path=None):
    """
    Plot hierarchical clustering dendrogram.
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    dendrogram(linkage_matrix, ax=ax, leaf_rotation=90, leaf_font_size=8)
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_clustering_comparison(X_2d, y_true, y_pred, save_path=None):
    """
    Compare true labels vs clustering results.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # True labels
    scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='Set1', alpha=0.6, s=50)
    axes[0].set_title('True Labels', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    
    # Predicted clusters
    scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap='Set1', alpha=0.6, s=50)
    axes[1].set_title('Clustering Results', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

