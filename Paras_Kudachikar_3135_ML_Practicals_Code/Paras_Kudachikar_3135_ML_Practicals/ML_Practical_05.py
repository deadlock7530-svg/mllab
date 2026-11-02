# Machine Learning Laboratory - Practical No. 05
# Name: Paras Kudachikar | Roll No.: 3135

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load Diabetes dataset
diabetes = load_diabetes(as_frame=True)

X = diabetes.data
y = diabetes.target

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Create models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge (alpha=1.0)": Ridge(alpha=1.0),
    "Lasso (alpha=2.0)": Lasso(alpha=2.0)
}

# Print table heading
print(f"{'Model':<20} | {'R2':>8} | {'RMSE':>8} | {'Non-zero coefs':>15}")
print("-" * 62)

results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    nz = np.sum(np.abs(model.coef_) > 1e-4)
    results[name] = model.coef_
    print(
        f"{name:<20} | {r2:>8.4f} | "
        f"{rmse:>8.3f} | {nz:>15}"
    )

# Display coefficient comparison
print("\nCoefficient comparison (feature shrinkage):")

print(
    f"{'Feature':<10}" +
    "".join(f"{name[:12]:>15}" for name in models)
)

for i, feature in enumerate(X.columns):
    row = f"{feature:<10}"
    for name in models:
        row += f"{results[name][i]:>15.2f}"
    print(row)

# Create bar chart
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(X.columns))
width = 0.25

for i, (name, coefficients) in enumerate(results.items()):
    ax.bar(
        x + i * width,
        coefficients,
        width,
        label=name
    )

ax.set_xticks(x + width)
ax.set_xticklabels(X.columns)
ax.set_ylabel("Coefficient value")
ax.set_title(
    "Ridge vs Lasso: Coefficient Shrinkage "
    "(Diabetes Dataset)"
)
ax.legend()
ax.axhline(
    0,
    linewidth=0.8
)
plt.tight_layout()

# Save graph
plt.savefig(
    "ridge_lasso_diabetes.png",
    dpi=130
)
plt.close()

# Observation
print("\nObservation:")
print(
    "Lasso drives several coefficients to exactly zero "
    "(feature selection),"
)
print(
    "while Ridge shrinks all coefficients but usually "
    "keeps them non-zero."
)
print(
    "Both methods help reduce the effect of "
    "multicollinearity."
)
print("\nGraph saved as: ridge_lasso_diabetes.png")
