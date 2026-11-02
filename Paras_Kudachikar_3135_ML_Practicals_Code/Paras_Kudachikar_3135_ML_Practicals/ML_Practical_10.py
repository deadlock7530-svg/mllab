# Machine Learning Laboratory - Practical No. 10
# Name: Paras Kudachikar | Roll No.: 3135

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    'Decision Tree (baseline)': DecisionTreeClassifier(random_state=42),
    'Random Forest (100 trees)': RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

print(f"{'Model':<28} | {'Accuracy':>9} | {'F1-score':>9}")
print("-" * 52)

accs = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    accs[name] = acc
    print(f"{name:<28} | {acc:>9.4f} | {f1:>9.4f}")

rf = models['Random Forest (100 trees)']
importances = rf.feature_importances_
idx = np.argsort(importances)[-10:]

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].barh(range(len(idx)), importances[idx])
axs[0].set_yticks(range(len(idx)))
axs[0].set_yticklabels([data.feature_names[i] for i in idx])
axs[0].set_xlabel('Feature Importance')
axs[0].set_title('Random Forest: Top 10 Important Features')

axs[1].bar(list(accs.keys()), list(accs.values()))
axs[1].set_ylabel('Accuracy')
axs[1].set_ylim(0.85, 1.0)
axs[1].set_title('Baseline vs Ensemble Comparison')
plt.setp(axs[1].get_xticklabels(), rotation=15, ha='right')
plt.tight_layout()
plt.savefig('a10_ensemble.png', dpi=130)

print("\nObservation: Both ensemble methods (Random Forest, Gradient Boosting)")
print("outperform the single Decision Tree baseline by reducing variance")
print("(bagging) / bias (boosting) respectively.")
