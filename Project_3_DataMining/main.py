# ===========================================
# Main Script - Project 3: Data Mining
# ===========================================

"""
Main script for running data mining experiments.
Includes EDA, clustering, dimensionality reduction, and anomaly detection.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_steel_data, get_class_names, get_feature_names
from preprocessing import preprocess_for_mining, compute_statistics
from mining import (perform_pca, perform_tsne, kmeans_clustering, find_optimal_k,
                   hierarchical_clustering, dbscan_clustering, detect_anomalies,
                   evaluate_clustering)
from visualization import (plot_correlation_heatmap, plot_class_distribution,
                          plot_pca_2d, plot_tsne, plot_elbow, plot_dendrogram,
                          plot_clustering_comparison)


def main():
    """Main function to run all data mining experiments."""
    
    print("="*60)
    print("PROJECT 3: DATA MINING")
    print("Pattern Discovery and Knowledge Extraction")
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
    X_scaled, y_encoded, scaler, encoder = preprocess_for_mining(X, y)
    
    # Step 3: Exploratory Data Analysis
    print("\n[3] Exploratory Data Analysis...")
    
    # Statistics
    stats = compute_statistics(X, feature_names)
    stats.to_csv('figures/descriptive_statistics.csv')
    print("Descriptive statistics saved.")
    
    # Correlation heatmap
    plot_correlation_heatmap(X, feature_names, save_path='figures/correlation_heatmap.png')
    
    # Class distribution
    plot_class_distribution(y, class_names, save_path='figures/class_distribution.png')
    
    # Step 4: Dimensionality Reduction
    print("\n[4] Dimensionality Reduction...")
    
    # PCA
    X_pca, pca, explained = perform_pca(X_scaled, n_components=2)
    plot_pca_2d(X_pca, y_encoded, class_names, save_path='figures/pca_2d.png')
    
    # t-SNE
    X_tsne = perform_tsne(X_scaled, n_components=2, perplexity=30)
    plot_tsne(X_tsne, y_encoded, class_names, save_path='figures/tsne.png')
    
    # Step 5: Clustering
    print("\n[5] Clustering Analysis...")
    
    # Find optimal K
    elbow_results, optimal_k = find_optimal_k(X_scaled, k_range=range(2, 12))
    elbow_results.to_csv('figures/elbow_analysis.csv', index=False)
    plot_elbow(elbow_results, save_path='figures/elbow_plot.png')
    
    # K-Means with optimal K
    kmeans_labels, kmeans_model, kmeans_silhouette = kmeans_clustering(X_scaled, n_clusters=7)
    kmeans_eval = evaluate_clustering(y_encoded, kmeans_labels)
    
    # Hierarchical clustering
    hc_labels, linkage_matrix, hc_silhouette = hierarchical_clustering(X_scaled, n_clusters=7)
    hc_eval = evaluate_clustering(y_encoded, hc_labels)
    
    # Dendrogram (on sample)
    sample_idx = np.random.choice(len(X_scaled), min(200, len(X_scaled)), replace=False)
    _, linkage_sample, _ = hierarchical_clustering(X_scaled[sample_idx], n_clusters=7)
    plot_dendrogram(linkage_sample, save_path='figures/dendrogram.png')
    
    # DBSCAN
    dbscan_labels, n_clusters, n_noise = dbscan_clustering(X_scaled, eps=2.0, min_samples=5)
    
    # Clustering comparison visualization
    plot_clustering_comparison(X_pca, y_encoded, kmeans_labels, 
                              save_path='figures/clustering_comparison.png')
    
    # Step 6: Anomaly Detection
    print("\n[6] Anomaly Detection...")
    anomaly_labels, anomaly_scores, n_anomalies = detect_anomalies(X_scaled, contamination=0.1)
    
    # Save anomaly results
    anomaly_df = pd.DataFrame({
        'Index': range(len(anomaly_labels)),
        'Label': anomaly_labels,
        'Score': anomaly_scores
    })
    anomaly_df.to_csv('figures/anomaly_results.csv', index=False)
    
    # Step 7: Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Clustering comparison table
    clustering_results = pd.DataFrame([
        {'Method': 'K-Means', 'Silhouette': kmeans_silhouette, 'ARI': kmeans_eval['ARI'], 'NMI': kmeans_eval['NMI']},
        {'Method': 'Hierarchical', 'Silhouette': hc_silhouette, 'ARI': hc_eval['ARI'], 'NMI': hc_eval['NMI']}
    ])
    clustering_results.to_csv('figures/clustering_comparison.csv', index=False)
    
    print("\nClustering Results:")
    print(clustering_results.to_string(index=False))
    
    print(f"\nPCA Explained Variance: {sum(explained):.4f}")
    print(f"Optimal K (by silhouette): {optimal_k}")
    print(f"Anomalies detected: {n_anomalies} ({n_anomalies/len(X_scaled)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("Data Mining experiments completed!")
    print("Results saved to 'figures/' directory")
    print("="*60)


if __name__ == "__main__":
    main()

