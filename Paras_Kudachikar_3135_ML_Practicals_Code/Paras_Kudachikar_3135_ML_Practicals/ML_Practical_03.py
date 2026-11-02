# Machine Learning Laboratory - Practical No. 03
# Name: Paras Kudachikar | Roll No.: 3135

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes(as_frame=True)
X = diabetes.data
y = diabetes.target

print("Features:", list(X.columns))
print("Dataset shape:", X.shape)
print(X.head(), "\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Coefficients:")
for f, c in zip(X.columns, model.coef_):
    print(f" {f:>6}: {c:8.3f}")

print(f"Intercept: {model.intercept_:.3f}\n")
print(f"MSE : {mse:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R2 : {r2:.4f}")

plt.figure(figsize=(6, 5))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7,
    color='#4C72B0',
    edgecolor='k'
)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    'r--',
    lw=2
)

plt.xlabel('Actual Disease Progression')
plt.ylabel('Predicted Disease Progression')
plt.title('Multiple Linear Regression: Actual vs Predicted')

plt.tight_layout()
os.makedirs('imgs', exist_ok=True)
plt.savefig('imgs/a3_mlr.png', dpi=130)

print("\nPlot saved.")
