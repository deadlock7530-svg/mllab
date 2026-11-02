import numpy as np 
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_moons 
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import DBSCAN, KMeans 
from sklearn.metrics import silhouette_score 
moon_features, true_labels = make_moons( 
n_samples=300, 
noise=0.08, 
random_state=42 
) 
scaled_features = StandardScaler().fit_transform(moon_features) 
 
dbscan_model = DBSCAN( 
    eps=0.25, 
    min_samples=5 
) 
 
cluster_labels = dbscan_model.fit_predict(scaled_features) 
 
cluster_count = len(set(cluster_labels)) - ( 
    1 if -1 in cluster_labels else 0 
) 
 
noise_count = list(cluster_labels).count(-1) 
 
print( 
    f"Estimated number of clusters: {cluster_count}" 
) 
 
print( 
    f"Estimated number of noise points: {noise_count}" 
) 
 
if cluster_count > 1: 
 
    silhouette_value = silhouette_score( 
        scaled_features, 
        cluster_labels 
    ) 
 
    print( 
        f"Silhouette Score: {silhouette_value:.4f}" 
    ) 
 
kmeans_model = KMeans( 
    n_clusters=2, 
    random_state=42, 
    n_init=10 
).fit(scaled_features) 
 
figure, axes = plt.subplots( 
    1, 
    2, 
    figsize=(11,4.5) 
) 
 
axes[0].scatter( 
    moon_features[:,0], 
    moon_features[:,1], 
    c=kmeans_model.labels_, 
    cmap='viridis', 
    s=25, 
    edgecolor='k' 
) 
 
axes[0].set_title( 
    'K-Means (fails on non-convex shapes)' 
) 
 
axes[1].scatter( 
    moon_features[:,0], 
    moon_features[:,1], 
    c=cluster_labels, 
    cmap='viridis', 
    s=25, 
    edgecolor='k' 
) 
 
noise_mask = cluster_labels == -1 
 
axes[1].scatter( 
    moon_features[noise_mask,0], 
    moon_features[noise_mask,1], 
    c='red', 
    marker='x', 
    s=40, 
    label='Noise' 
) 
 
axes[1].set_title( 
    f'DBSCAN (clusters={cluster_count}, noise={noise_count})' 
) 
 
axes[1].legend() 
 
plt.tight_layout() 
 
plt.savefig( 
    'imgs/a9_dbscan.png', 
    dpi=130 
) 
print( 
"\nObservation: DBSCAN correctly separates the two crescent " 
"(non-convex) clusters and" 
) 
print( 
"flags outliers as noise, while K-Means " 
"(assumes spherical clusters) fails on this shape." 
) 