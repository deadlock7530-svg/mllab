import numpy as np 
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_diabetes 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error, r2_score 
diabetes_data = load_diabetes(as_frame=True) 
features, target = diabetes_data.data, diabetes_data.target 
features_train, features_test, target_train, target_test = train_test_split( 
    features, 
    target, 
    test_size=0.2, 
    random_state=42 
) 
 
standard_scaler = StandardScaler() 
 
features_train_scaled = standard_scaler.fit_transform(features_train) 
features_test_scaled = standard_scaler.transform(features_test) 
 
regression_models = { 
    'Linear Regression': LinearRegression(), 
    'Ridge (alpha=1.0)': Ridge(alpha=1.0), 
    'Lasso (alpha=2.0)': Lasso(alpha=2.0) 
} 
 
print( 
    f"{'Model':<20} | {'R2':>8} | {'RMSE':>8} | " 
    f"{'Non-zero coefs':>15}" 
) 
 
print("-"*62) 
 
model_results = {} 
 
for model_name, regression_model in regression_models.items(): 
 
    regression_model.fit(features_train_scaled, target_train) 
    predicted_values = regression_model.predict(features_test_scaled) 
 
    r_squared = r2_score(target_test, predicted_values) 
 
    root_mean_squared_error = np.sqrt( 
        mean_squared_error(target_test, predicted_values) 
    ) 
 
    non_zero_coefficients = np.sum( 
        np.abs(regression_model.coef_) > 1e-4 
    ) 
 
    model_results[model_name] = regression_model.coef_ 
 
    print( 
        f"{model_name:<20} | " 
        f"{r_squared:>8.4f} | " 
        f"{root_mean_squared_error:>8.3f} | " 
        f"{non_zero_coefficients:>15}" 
    ) 
 
print("\nCoefficient comparison (feature shrinkage):") 
 
print( 
    f"{'Feature':<10}" + 
    "".join( 
        f"{model_name[:12]:>15}" 
        for model_name in regression_models 
    ) 
) 
for feature_index, feature_name in enumerate(features.columns): 
 
    coefficient_row = f"{feature_name:<10}" 
 
    for model_name in regression_models: 
        coefficient_row += ( 
            f"{model_results[model_name][feature_index]:>15.2f}" 
        ) 
 
    print(coefficient_row) 
 
figure, axes = plt.subplots(figsize=(9,5)) 
 
x_positions = np.arange(len(features.columns)) 
bar_width = 0.25 
 
for model_index, (model_name, coefficients) in enumerate( 
    model_results.items() 
): 
 
    axes.bar( 
        x_positions + model_index * bar_width, 
        coefficients, 
        bar_width, 
        label=model_name 
    ) 
 
axes.set_xticks(x_positions + bar_width) 
axes.set_xticklabels(features.columns) 
axes.set_ylabel('Coefficient value') 
axes.set_title( 
'Ridge vs Lasso: Coefficient Shrinkage (Diabetes Dataset)' 
) 
axes.legend() 
axes.axhline(0, color='black', linewidth=0.8) 
plt.tight_layout() 
plt.savefig('imgs/a5_ridge_lasso.png', dpi=130) 
print( 
"\nObservation: Lasso drives several coefficients to exactly zero " 
"(feature selection)," 
) 
print( 
"while Ridge shrinks all coefficients but keeps them non-zero, " 
"reducing multicollinearity impact." 
)