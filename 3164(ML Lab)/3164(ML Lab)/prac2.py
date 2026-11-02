import numpy as np 
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_wine 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
import time 
 
wine = load_wine() 
X, y = wine.data, wine.target 
print("Dataset shape:", X.shape, " Classes:", np.unique(y)) 
 
X_scaled = StandardScaler().fit_transform(X) 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

t0 = time.time() 
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train) 
acc_full = accuracy_score(y_test, clf.predict(X_test)) 
t_full = time.time() - t0

pca = PCA(n_components=2) 
X_train_pca = pca.fit_transform(X_train) 
X_test_pca = pca.transform(X_test) 
t0 = time.time() 
clf_pca = LogisticRegression(max_iter=1000).fit(X_train_pca, y_train) 
acc_pca = accuracy_score(y_test, clf_pca.predict(X_test_pca)) 
t_pca = time.time() - t0 
print(f"\nPCA Explained Variance Ratio (2 components): {pca.explained_variance_ratio_}") 
print(f"Total variance retained: {sum(pca.explained_variance_ratio_)*100:.2f}%") 

lda = LinearDiscriminantAnalysis(n_components=2) 
X_train_lda = lda.fit_transform(X_train, y_train) 
X_test_lda = lda.transform(X_test) 
t0 = time.time() 
clf_lda = LogisticRegression(max_iter=1000).fit(X_train_lda, y_train) 
acc_lda = accuracy_score(y_test, clf_lda.predict(X_test_lda)) 
t_lda = time.time() - t0

print(f"\nAccuracy with all 13 features : {acc_full:.4f}  (train time: {t_full*1000:.2f} ms)") 
print(f"Accuracy with PCA (2 comp)    : {acc_pca:.4f}  (train time: {t_pca*1000:.2f} ms)") 
print(f"Accuracy with LDA (2 comp)    : {acc_lda:.4f}  (train time: {t_lda*1000:.2f} ms)") 

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))

for target, color, label in zip(
    [0, 1, 2],
    ['#4C72B0', '#DD8452', '#55A868'],
    wine.target_names
):
    axs[0].scatter(
        X_train_pca[y_train == target, 0],
        X_train_pca[y_train == target, 1],
        c=color,
        label=label,
        s=25
    )

    axs[1].scatter(
        X_train_lda[y_train == target, 0],
        X_train_lda[y_train == target, 1],
        c=color,
        label=label,
        s=25
    )

axs[0].set_title('PCA - Wine Dataset (2 Components)')
axs[0].set_xlabel('PC1')
axs[0].set_ylabel('PC2')
axs[0].legend()

axs[1].set_title('LDA - Wine Dataset (2 Components)')
axs[1].set_xlabel('LD1')
axs[1].set_ylabel('LD2')
axs[1].legend()

plt.tight_layout()
plt.savefig('a2_pca_lda.png', dpi=130)
print("\nPlot saved.")