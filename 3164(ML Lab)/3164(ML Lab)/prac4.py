import numpy as np 
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score 
np.random.seed(42) 
input_values = np.sort(np.random.rand(80, 1) * 10, axis=0) 
output_values = ( 
    0.5 * input_values.ravel()**2 
    - 2 * input_values.ravel() 
    + 5 
    + np.random.randn(80) * 3 
) 
 
input_train, input_test, output_train, output_test = train_test_split( 
    input_values, 
    output_values, 
    test_size=0.2, 
    random_state=42 
) 
 
polynomial_degrees = [1, 2, 4, 8] 
 
plt.figure(figsize=(7,5)) 
 
plt.scatter( 
    input_train, 
    output_train, 
    color='gray', 
    s=20, 
    label='Train data', 
    alpha=0.6 
) 
 
input_plot = np.linspace(0, 10, 200).reshape(-1,1) 
 
print(f"{'Degree':>7} | {'Train R2':>9} | {'Test R2':>8} | {'Test RMSE':>10}") 
print("-"*45) 
for degree_value in polynomial_degrees: 
 
    polynomial_model = PolynomialFeatures(degree=degree_value) 
 
    input_train_polynomial = polynomial_model.fit_transform(input_train) 
    input_test_polynomial = polynomial_model.transform(input_test) 
 
    regression_model = LinearRegression().fit( 
        input_train_polynomial, 
        output_train 
    ) 
 
    train_r_squared = r2_score( 
        output_train, 
        regression_model.predict(input_train_polynomial) 
    ) 
 
    predicted_test_values = regression_model.predict(input_test_polynomial) 
 
    test_r_squared = r2_score( 
        output_test, 
        predicted_test_values 
    ) 
 
    test_root_mean_squared_error = np.sqrt( 
        mean_squared_error(output_test, predicted_test_values) 
    ) 
 
    print( 
        f"{degree_value:>7} | " 
        f"{train_r_squared:>9.4f} | " 
        f"{test_r_squared:>8.4f} | " 
        f"{test_root_mean_squared_error:>10.3f}" 
    ) 
 
    predicted_plot_values = regression_model.predict( 
        polynomial_model.transform(input_plot) 
    ) 
 
    plt.plot( 
        input_plot, 
        predicted_plot_values, 
        label=f'Degree {degree_value}', 
        linewidth=2 
    ) 
 
plt.xlabel('X') 
plt.ylabel('y') 
plt.title('Polynomial Regression: Bias-Variance Trade-off') 
plt.legend() 
plt.tight_layout() 
 
plt.savefig('imgs/a4_poly.png', dpi=130) 
 
print( 
    "\nObservation: Degree 1 underfits (high bias); " 
    "Degree 8 starts overfitting (high variance," 
) 
 
print( 
"test R2 drops relative to degree 2-4 despite higher train R2)." 
)