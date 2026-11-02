# Machine Learning Laboratory - Practical No. 04
# Name: Paras Kudachikar | Roll No.: 3135

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
#--- 2. CREATE SYNTHETIC NON-LINEAR DATASET ---
np.random.seed(42)
X = np.linspace(-3, 3, 80).reshape(-1, 1)
y = 2 * X[:, 0]**2 + 3 * X[:, 0] + 5
y = y + np.random.normal(0, 3, 80)
#--- 3. SPLIT DATASET ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#--- 4. TRY DIFFERENT POLYNOMIAL DEGREES ---
degrees = [1, 2, 4, 8]
plt.figure(figsize=(12, 8))
for i, degree in enumerate(degrees):
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    # Create model
    model = LinearRegression()
    # Train model
    model.fit(X_train_poly, y_train)
    # Predictions
    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)
    # Performance
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    print("\nDegree:", degree)
    print("Train R2 :", round(train_r2, 4))
    print("Test R2 :", round(test_r2, 4))
    print("Train RMSE :", round(train_rmse, 4))
    print("Test RMSE :", round(test_rmse, 4))
    #--- 5. PLOT FITTED CURVE ---
    X_curve = np.linspace(-3, 3, 300).reshape(-1, 1)
    X_curve_poly = poly.transform(X_curve)
    y_curve = model.predict(X_curve_poly)
    plt.subplot(2, 2, i + 1)
    plt.scatter(X, y, color="black", s=20)
    plt.plot(X_curve, y_curve, color="red")
    plt.title("Polynomial Degree " + str(degree))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
plt.tight_layout()
plt.show()
