# Machine Learning Laboratory - Practical No. 08
# Name: Paras Kudachikar | Roll No.: 3135

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))  # silences the joblib/loky "physical cores" warning on Windows

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Load Iris dataset
iris = load_iris()
X = iris.data
# Standardization
X_scaled = StandardScaler().fit_transform(X)
wcss = []
sil_scores = []
K_range = range(2, 9)
# Calculate WCSS and Silhouette Score
for k in K_range:
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)
    sil_scores.append(
        silhouette_score(
            X_scaled,
            labels
        )
    )
# Display results
print(
    f"{'k':>3} | "
    f"{'WCSS (Inertia)':>15} | "
    f"{'Silhouette Score':>17}"
)
print("-" * 40)
for k, w, s in zip(
    K_range,
    wcss,
    sil_scores
):
    print(
        f"{k:>3} | "
        f"{w:>15.3f} | "
        f"{s:>17.4f}"
    )
# Find optimal k
best_k = list(K_range)[
    np.argmax(sil_scores)
]
print(
    f"\nOptimal k (highest silhouette score) = {best_k}"
)
# Final K-Means model
km_final = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)
labels_final = km_final.fit_predict(X_scaled)
# Create plots
fig, axs = plt.subplots(
    1,
    3,
    figsize=(15, 4.5)
)
# Elbow Method
axs[0].plot(
    list(K_range),
    wcss,
    'o-',
    color='#4C72B0'
)
axs[0].set_xlabel(
    'Number of clusters (k)'
)
axs[0].set_ylabel('WCSS')
axs[0].set_title('Elbow Method')
# Silhouette Method
axs[1].plot(
    list(K_range),
    sil_scores,
    'o-',
    color='#DD8452'
)
axs[1].set_xlabel(
    'Number of clusters (k)'
)
axs[1].set_ylabel(
    'Silhouette Score'
)
axs[1].set_title(
    'Silhouette Method'
)
# Final clusters
axs[2].scatter(
    X[:, 2],
    X[:, 3],
    c=labels_final,
    cmap='viridis',
    s=30,
    edgecolor='k'
)
axs[2].set_xlabel(
    'Petal length'
)
axs[2].set_ylabel(
    'Petal width'
)
axs[2].set_title(
    f'K-Means Clusters (k={best_k})'
)
plt.tight_layout()
plt.savefig(
    "practical_8_kmeans.png",
    dpi=130
)
print("\nPlot saved as practical_8_kmeans.png")
