# Machine Learning Laboratory - Practical No. 07
# Name: Paras Kudachikar | Roll No.: 3135

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
# Load Iris dataset
iris = load_iris()
# Petal length and petal width
X = iris.data[:, [2, 3]]
y = iris.target
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
# Feature scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
# SVM kernels
kernels = ['linear', 'rbf', 'poly']
results = {}
print(f"{'Kernel':<10} | {'Accuracy':>9}")
print("-" * 23)
for k in kernels:
    clf = SVC(
        kernel=k,
        C=1.0,
        gamma='scale'
    )
    clf.fit(X_train_s, y_train)
    acc = accuracy_score(
        y_test,
        clf.predict(X_test_s)
    )
    results[k] = (clf, acc)
    print(f"{k:<10} | {acc:>9.4f}")
# Create decision boundary plots
fig, axs = plt.subplots(
    1,
    3,
    figsize=(15, 4.5)
)
xx, yy = np.meshgrid(
    np.linspace(
        X_train_s[:, 0].min() - 1,
        X_train_s[:, 0].max() + 1,
        200
    ),
    np.linspace(
        X_train_s[:, 1].min() - 1,
        X_train_s[:, 1].max() + 1,
        200
    )
)
for ax, k in zip(axs, kernels):
    clf, acc = results[k]
    Z = clf.predict(
        np.c_[xx.ravel(), yy.ravel()]
    ).reshape(xx.shape)
    ax.contourf(
        xx,
        yy,
        Z,
        alpha=0.3,
        cmap='viridis'
    )
    ax.scatter(
        X_train_s[:, 0],
        X_train_s[:, 1],
        c=y_train,
        cmap='viridis',
        edgecolor='k',
        s=25
    )
    ax.set_title(
        f"SVM ({k} kernel)\nAccuracy: {acc:.2f}"
    )
    ax.set_xlabel('Petal length (scaled)')
    ax.set_ylabel('Petal width (scaled)')
plt.tight_layout()
plt.savefig(
    "practical_7_svm.png",
    dpi=130
)
print("\nBest kernel:",
      max(results, key=lambda k: results[k][1]))
print(
    "\nClassification Report (RBF kernel):\n",
    classification_report(
        y_test,
        results['rbf'][0].predict(X_test_s),
        target_names=iris.target_names
    )
)
print("Plot saved as practical_7_svm.png")
