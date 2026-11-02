# Machine Learning Laboratory - Practical No. 02
# Name: Paras Kudachikar | Roll No.: 3135

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

# Load Wine dataset
wine = load_wine()
X, y = wine.data, wine.target

print("Dataset Shape:", X.shape)
print("Classes:", np.unique(y))

# Standardization
X = StandardScaler().fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------- Full Features ----------------
start = time.time()

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

acc_full = accuracy_score(y_test, model.predict(X_test))
time_full = (time.time() - start) * 1000

# ---------------- PCA ----------------
pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

start = time.time()

model_pca = LogisticRegression(max_iter=1000)
model_pca.fit(X_train_pca, y_train)

acc_pca = accuracy_score(y_test, model_pca.predict(X_test_pca))
time_pca = (time.time() - start) * 1000

print("\nPCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("Total Variance Retained:",
      round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

# ---------------- LDA ----------------
lda = LinearDiscriminantAnalysis(n_components=2)

X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

start = time.time()

model_lda = LogisticRegression(max_iter=1000)
model_lda.fit(X_train_lda, y_train)

acc_lda = accuracy_score(y_test, model_lda.predict(X_test_lda))
time_lda = (time.time() - start) * 1000

# ---------------- Results ----------------
print("\n--- RESULTS ---")

print("Full Features (13): Accuracy =", round(acc_full, 4),
      "Time =", round(time_full, 2), "ms")

print("PCA (2 features): Accuracy =", round(acc_pca, 4),
      "Time =", round(time_pca, 2), "ms")

print("LDA (2 features): Accuracy =", round(acc_lda, 4),
      "Time =", round(time_lda, 2), "ms")

# ---------------- Graph ----------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train)
plt.title("PCA - 2 Components")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.subplot(1, 2, 2)
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train)
plt.title("LDA - 2 Components")
plt.xlabel("LD1")
plt.ylabel("LD2")

plt.tight_layout()
plt.show()
