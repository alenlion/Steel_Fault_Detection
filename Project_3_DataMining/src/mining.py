# ===========================================
# Mining Module - Project 3: Data Mining
# ===========================================

"""
Module for data mining techniques: clustering, dimensionality reduction, anomaly detection.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage


def perform_pca(X, n_components=2):
    """
    Perform PCA dimensionality reduction.
    """
    
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    
    explained = pca.explained_variance_ratio_
    print(f"PCA with {n_components} components:")
    print(f"  - Total explained variance: {sum(explained):.4f}")
    
    return X_pca, pca, explained


def perform_tsne(X, n_components=2, perplexity=30):
    """
    Perform t-SNE dimensionality reduction.
    """
    
    print(f"Performing t-SNE (perplexity={perplexity})...")
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    print("t-SNE completed!")
    
    return X_tsne


def kmeans_clustering(X, n_clusters=7):
    """
    Perform K-Means clustering.
    """
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    silhouette = silhouette_score(X, labels)
    inertia = kmeans.inertia_
    
    print(f"K-Means (k={n_clusters}):")
    print(f"  - Silhouette score: {silhouette:.4f}")
    print(f"  - Inertia: {inertia:.2f}")
    
    return labels, kmeans, silhouette


def find_optimal_k(X, k_range=range(2, 11)):
    """
    Find optimal K using elbow method and silhouette score.
    """
    
    results = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        silhouette = silhouette_score(X, labels)
        inertia = kmeans.inertia_
        
        results.append({
            'k': k,
            'silhouette': silhouette,
            'inertia': inertia
        })
    
    results_df = pd.DataFrame(results)
    best_k = results_df.loc[results_df['silhouette'].idxmax(), 'k']
    
    print(f"Optimal k (silhouette): {best_k}")
    
    return results_df, int(best_k)


def hierarchical_clustering(X, n_clusters=7, method='ward'):
    """
    Perform hierarchical clustering.
    """
    
    linkage_matrix = linkage(X, method=method)
    
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    labels = hc.fit_predict(X)
    
    silhouette = silhouette_score(X, labels)
    
    print(f"Hierarchical Clustering (method={method}):")
    print(f"  - Silhouette score: {silhouette:.4f}")
    
    return labels, linkage_matrix, silhouette


def dbscan_clustering(X, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering.
    """
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"DBSCAN (eps={eps}, min_samples={min_samples}):")
    print(f"  - Clusters found: {n_clusters}")
    print(f"  - Noise points: {n_noise}")
    
    return labels, n_clusters, n_noise


def detect_anomalies(X, contamination=0.1):
    """
    Detect anomalies using Isolation Forest.
    """
    
    iso = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(X)
    scores = iso.decision_function(X)
    
    n_anomalies = (labels == -1).sum()
    
    print(f"Anomaly Detection (contamination={contamination}):")
    print(f"  - Anomalies found: {n_anomalies} ({n_anomalies/len(X)*100:.1f}%)")
    
    return labels, scores, n_anomalies


def evaluate_clustering(y_true, y_pred):
    """
    Evaluate clustering against true labels.
    """
    
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    
    print("Clustering Evaluation:")
    print(f"  - Adjusted Rand Index: {ari:.4f}")
    print(f"  - Normalized Mutual Info: {nmi:.4f}")
    
    return {'ARI': ari, 'NMI': nmi}

