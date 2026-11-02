# Machine Learning Laboratory - Practical No. 09
# Name: Paras Kudachikar | Roll No.: 3135

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))  # silences the joblib/loky "physical cores" warning on Windows

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

X, y_true = make_moons(n_samples=300, noise=0.08, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps=0.25, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)

print(f"Estimated number of clusters: {n_clusters}")
print(f"Estimated number of noise points: {n_noise}")
if n_clusters > 1:
    sil = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {sil:.4f}")

km = KMeans(n_clusters=2, random_state=42, n_init=10)
km.fit(X_scaled)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))

axs[0].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='viridis', s=25, edgecolor='k')

axs[0].set_title('K-Means (fails on non-convex shapes)')
axs[0].set_xlabel('X1')
axs[0].set_ylabel('X2')

axs[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=25, edgecolor='k')

noise_mask = labels == -1
axs[1].scatter(X[noise_mask, 0], X[noise_mask, 1], marker='x', s=40, label='Noise')
axs[1].set_title(f'DBSCAN (clusters={n_clusters}, noise={n_noise})')
axs[1].set_xlabel('X1')
axs[1].set_ylabel('X2')

axs[1].legend()

plt.tight_layout()
plt.savefig('a9_dbscan.png', dpi=130)

print("\nObservation: DBSCAN correctly separates the two crescent "
      "(non-convex) clusters and flags outliers as noise, while "
      "K-Means assumes spherical clusters and fails on this shape.")
