# Machine Learning Laboratory - Practical No. 06
# Name: Paras Kudachikar | Roll No.: 3135

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report
)

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

print("Dataset shape:", X.shape, " Classes:", data.target_names)

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

# Logistic Regression model
model = LogisticRegression(max_iter=5000)
model.fit(X_train_s, y_train)

# Prediction
y_pred = model.predict(X_test_s)

# Evaluation
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nConfusion Matrix:\n", cm)
print(f"\nAccuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

print(
    "\nClassification Report:\n",
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(5, 4.5))

im = ax.imshow(cm, cmap='Blues')

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            cm[i, j],
            ha='center',
            va='center',
            color='black',
            fontsize=14
        )

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(data.target_names)
ax.set_yticklabels(data.target_names)

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

ax.set_title(
    'Confusion Matrix - Logistic Regression\n'
    '(Breast Cancer Dataset)'
)

plt.colorbar(im)
plt.tight_layout()
plt.savefig(
    "practical_6_confusion_matrix.png",
    dpi=130
)
print("\nPlot saved as practical_6_confusion_matrix.png")
