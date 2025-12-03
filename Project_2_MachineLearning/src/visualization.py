# ===========================================
# Visualization Module - Project 2: Machine Learning
# ===========================================

"""
Module for creating visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_model_comparison(results_df, metric='Accuracy', save_path=None):
    """
    Plot model comparison bar chart.
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    results_sorted = results_df.sort_values(metric, ascending=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_sorted)))
    
    bars = ax.barh(results_sorted['Model'], results_sorted[metric], color=colors)
    
    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, results_sorted[metric]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_confusion_matrix(cm, class_names=None, normalize=False, save_path=None):
    """
    Plot confusion matrix heatmap.
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_roc_curves(models, X_test, y_test, class_names=None, save_path=None):
    """
    Plot ROC curves for multiple models.
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            
            # Compute micro-average ROC
            y_bin = label_binarize(y_test, classes=np.unique(y_test))
            n_classes = y_bin.shape[1]
            
            fpr, tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (Micro-Average)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_feature_importance(importance_df, top_n=15, save_path=None):
    """
    Plot feature importance.
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = importance_df.head(top_n)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
    
    ax.barh(top_features['Feature'], top_features['Importance'], color=colors)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_metrics_radar(results_df, models_to_plot=None, save_path=None):
    """
    Plot radar chart comparing model metrics.
    """
    
    if models_to_plot:
        df = results_df[results_df['Model'].isin(models_to_plot)]
    else:
        df = results_df.head(5)  # Top 5 models
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    
    for (_, row), color in zip(df.iterrows(), colors):
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_title('Model Metrics Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

