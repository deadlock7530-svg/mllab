import numpy as np 
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris 
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score 
iris_data = load_iris() 
features = iris_data.data 
scaled_features = StandardScaler().fit_transform(features) 
wcss_values = [] 
silhouette_scores = [] 
 
cluster_range = range(2, 9) 
 
for cluster_count in cluster_range: 
 
    kmeans_model = KMeans( 
        n_clusters=cluster_count, 
        random_state=42, 
        n_init=10 
    ) 
 
    cluster_labels = kmeans_model.fit_predict(scaled_features) 
 
    wcss_values.append(kmeans_model.inertia_) 
 
    silhouette_scores.append( 
        silhouette_score(scaled_features, cluster_labels) 
    ) 
 
print( 
    f"{'k':>3} | {'WCSS (Inertia)':>15} | " 
    f"{'Silhouette Score':>17}" 
) 
 
print("-"*40) 
 
for cluster_count, wcss_value, silhouette_value in zip( 
    cluster_range, 
    wcss_values, 
    silhouette_scores 
): 
    print( 
        f"{cluster_count:>3} | " 
        f"{wcss_value:>15.3f} | " 
        f"{silhouette_value:>17.4f}" 
    ) 
 
optimal_clusters = list(cluster_range)[ 
    np.argmax(silhouette_scores) 
] 
 
print( 
    f"\nOptimal k (highest silhouette score) = " 
    f"{optimal_clusters}" 
) 
 
final_kmeans_model = KMeans( 
    n_clusters=optimal_clusters, 
    random_state=42, 
    n_init=10 
) 
 
final_cluster_labels = final_kmeans_model.fit_predict( 
    scaled_features 
) 
 
figure, axes = plt.subplots( 
    1, 
    3, 
    figsize=(15,4.5) 
) 
 
axes[0].plot( 
    list(cluster_range), 
    wcss_values, 
    'o-', 
    color='#4C72B0' 
) 
 
axes[0].set_xlabel('Number of clusters (k)') 
axes[0].set_ylabel('WCSS') 
axes[0].set_title('Elbow Method') 
 
axes[1].plot( 
    list(cluster_range), 
    silhouette_scores, 
    'o-', 
    color='#DD8452' 
) 
 
axes[1].set_xlabel('Number of clusters (k)') 
axes[1].set_ylabel('Silhouette Score') 
axes[1].set_title('Silhouette Method') 
 
scatter_plot = axes[2].scatter( 
    features[:,2], 
    features[:,3], 
    c=final_cluster_labels, 
    cmap='viridis', 
    s=30, 
    edgecolor='k' 
) 
 
axes[2].set_xlabel('Petal length') 
axes[2].set_ylabel('Petal width') 
axes[2].set_title( 
    f'K-Means Clusters (k={optimal_clusters})' 
) 
 
plt.tight_layout() 
plt.savefig( 
    'imgs/a8_kmeans.png', 
    dpi=130 
) 
 
print("\nPlot saved.") 