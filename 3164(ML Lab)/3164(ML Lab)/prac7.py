import numpy as np 
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, classification_report 
iris_data = load_iris() 
features = iris_data.data[:, [2,3]]  # petal length & width for 2D visualization 
target = iris_data.target 
features_train, features_test, target_train, target_test = train_test_split( 
    features, 
    target, 
    test_size=0.25, 
    random_state=42, 
    stratify=target 
) 
 
standard_scaler = StandardScaler() 
 
features_train_scaled = standard_scaler.fit_transform(features_train) 
features_test_scaled = standard_scaler.transform(features_test) 
 
kernel_types = ['linear', 'rbf', 'poly'] 
 
model_results = {} 
 
print(f"{'Kernel':<10} | {'Accuracy':>9}") 
print("-"*23) 
 
for kernel_type in kernel_types: 
 
    svm_model = SVC( 
        kernel=kernel_type, 
        C=1.0, 
        gamma='scale' 
    ) 
 
    svm_model.fit(features_train_scaled, target_train) 
 
    accuracy_value = accuracy_score( 
        target_test, 
        svm_model.predict(features_test_scaled) 
    ) 
 
    model_results[kernel_type] = ( 
        svm_model, 
        accuracy_value 
    ) 
 
    print( 
        f"{kernel_type:<10} | {accuracy_value:>9.4f}" 
    ) 
 
figure, axes = plt.subplots( 
    1, 
    3, 
    figsize=(15,4.5) 
) 
 
grid_x, grid_y = np.meshgrid( 
    np.linspace( 
        features_train_scaled[:,0].min()-1, 
        features_train_scaled[:,0].max()+1, 
        200 
    ), 
    np.linspace( 
        features_train_scaled[:,1].min()-1, 
        features_train_scaled[:,1].max()+1, 
        200 
    ) 
) 
 
for plot_axis, kernel_type in zip(axes, kernel_types): 
 
    svm_model, accuracy_value = model_results[kernel_type] 
 
    predicted_grid = svm_model.predict( 
        np.c_[grid_x.ravel(), grid_y.ravel()] 
    ).reshape(grid_x.shape) 
 
    plot_axis.contourf( 
        grid_x, 
        grid_y, 
        predicted_grid, 
        alpha=0.3, 
        cmap='viridis' 
    ) 
 
    plot_axis.scatter( 
        features_train_scaled[:,0], 
        features_train_scaled[:,1], 
        c=target_train, 
        cmap='viridis', 
        edgecolor='k', 
        s=25 
    ) 
 
    plot_axis.set_title( 
        f'SVM ({kernel_type} kernel)\n' 
        f'Accuracy: {accuracy_value:.2f}' 
    ) 
 
    plot_axis.set_xlabel('Petal length (scaled)') 
    plot_axis.set_ylabel('Petal width (scaled)') 
 
plt.tight_layout() 
 
plt.savefig( 
    'imgs/a7_svm.png', 
    dpi=130 
) 
 
print( 
    "\nBest kernel:", 
    max(model_results, key=lambda kernel_type: model_results[kernel_type][1]) 
) 
 
print( 
    "\nClassification Report (RBF kernel):\n", 
    classification_report( 
        target_test, 
        model_results['rbf'][0].predict(features_test_scaled), 
        target_names=iris_data.target_names 
    ) 
)