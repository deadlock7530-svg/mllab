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
cancer_data = load_breast_cancer() 
features, target = cancer_data.data, cancer_data.target 
print( 
"Dataset shape:", features.shape, 
" Classes:", cancer_data.target_names 
) 
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
logistic_model = LogisticRegression(max_iter=5000) 
logistic_model.fit(features_train_scaled, target_train) 
predicted_values = logistic_model.predict(features_test_scaled) 
confusion_matrix_values = confusion_matrix( 
    target_test, 
    predicted_values 
) 
 
accuracy_value = accuracy_score(target_test, predicted_values) 
 
precision_value = precision_score( 
    target_test, 
    predicted_values 
) 
 
recall_value = recall_score( 
    target_test, 
    predicted_values 
) 
 
f1_value = f1_score( 
    target_test, 
    predicted_values 
) 
 
print("\nConfusion Matrix:\n", confusion_matrix_values) 
 
print(f"\nAccuracy: {accuracy_value:.4f}") 
print(f"Precision: {precision_value:.4f}") 
print(f"Recall: {recall_value:.4f}") 
print(f"F1-score: {f1_value:.4f}") 
 
print( 
    "\nClassification Report:\n", 
    classification_report( 
        target_test, 
        predicted_values, 
        target_names=cancer_data.target_names 
    ) 
) 
 
figure, axes = plt.subplots(figsize=(5,4.5)) 
 
image = axes.imshow( 
    confusion_matrix_values, 
    cmap='Blues' 
) 
 
for row_index in range(2): 
    for column_index in range(2): 
        axes.text( 
            column_index, 
            row_index, 
            confusion_matrix_values[row_index, column_index], 
            ha='center', 
            va='center', 
            color='black', 
            fontsize=14 
        ) 
 
axes.set_xticks([0,1]) 
axes.set_yticks([0,1]) 
 
axes.set_xticklabels(cancer_data.target_names) 
axes.set_yticklabels(cancer_data.target_names) 
axes.set_xlabel('Predicted') 
axes.set_ylabel('Actual') 
axes.set_title( 
'Confusion Matrix - Logistic Regression\n' 
'(Breast Cancer Dataset)' 
) 
plt.colorbar(image) 
plt.tight_layout() 
plt.savefig( 
'imgs/a6_logreg_cm.png', 
dpi=130 
) 
print("\nPlot saved.")