"""
Clustering models
"""
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings("ignore")


def get_clusterers(n_clusters: int = 3, random_state: int = 42) -> Dict[str, Any]:
    return {
        "K-Means": KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    }


def evaluate_clustering(X, labels) -> Dict[str, Any]:
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    result = {
        "n_clusters": int(n_clusters),
        "cluster_sizes": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
    }
    try:
        if n_clusters > 1 and len(set(labels)) > 1:
            result["silhouette"] = float(silhouette_score(X, labels))
        else:
            result["silhouette"] = None
    except Exception:
        result["silhouette"] = None

    # 2D projection for visualization
    try:
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X)
        else:
            coords = X[:, :2] if X.shape[1] >= 2 else np.column_stack([X[:, 0], np.zeros(len(X))])
        result["projection"] = {
            "x": coords[:, 0].tolist()[:300],
            "y": coords[:, 1].tolist()[:300],
            "labels": labels.tolist()[:300],
        }
    except Exception:
        result["projection"] = None
    return result
