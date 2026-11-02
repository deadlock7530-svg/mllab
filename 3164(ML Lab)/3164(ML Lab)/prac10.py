import numpy as np 
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.metrics import accuracy_score, f1_score 
cancer_data = load_breast_cancer() 
features, target = cancer_data.data, cancer_data.target 
features_train, features_test, target_train, target_test = train_test_split( 
    features, 
    target, 
    test_size=0.25, 
    random_state=42, 
    stratify=target 
) 
 
classification_models = { 
    'Decision Tree (baseline)': DecisionTreeClassifier(random_state=42), 
    'Random Forest (100 trees)': RandomForestClassifier( 
        n_estimators=100, 
        random_state=42 
    ), 
    'Gradient Boosting': GradientBoostingClassifier( 
        random_state=42 
    ) 
} 
 
print( 
    f"{'Model':<28} | {'Accuracy':>9} | {'F1-score':>9}" 
) 
 
print("-"*52) 
 
accuracy_results = {} 
 
for model_name, classification_model in classification_models.items(): 
 
    classification_model.fit( 
        features_train, 
        target_train 
    ) 
 
    predicted_labels = classification_model.predict( 
        features_test 
    ) 
 
    accuracy_value = accuracy_score( 
        target_test, 
        predicted_labels 
    ) 
 
    f1_value = f1_score( 
        target_test, 
        predicted_labels 
    ) 
 
    accuracy_results[model_name] = accuracy_value 
 
    print( 
        f"{model_name:<28} | " 
        f"{accuracy_value:>9.4f} | " 
        f"{f1_value:>9.4f}" 
    ) 
 
random_forest_model = classification_models[ 
    'Random Forest (100 trees)' 
] 
 
feature_importance_values = random_forest_model.feature_importances_ 
 
important_feature_indices = np.argsort( 
    feature_importance_values 
)[-10:] 
 
figure, axes = plt.subplots( 
    1, 
    2, 
    figsize=(12,5) 
) 
 
axes[0].barh( 
    range(len(important_feature_indices)), 
    feature_importance_values[important_feature_indices], 
    color='#55A868' 
) 
 
axes[0].set_yticks( 
    range(len(important_feature_indices)) 
) 
 
axes[0].set_yticklabels( 
    [ 
        cancer_data.feature_names[index] 
        for index in important_feature_indices 
    ] 
) 
 
axes[0].set_xlabel('Feature Importance') 
axes[0].set_title( 
    'Random Forest: Top 10 Important Features' 
) 
 
axes[1].bar( 
    list(accuracy_results.keys()), 
    list(accuracy_results.values()), 
    color=['#C44E52','#4C72B0','#DD8452'] 
) 
 
axes[1].set_ylabel('Accuracy') 
axes[1].set_ylim(0.85,1.0) 
 
axes[1].set_title( 
    'Baseline vs Ensemble Comparison' 
) 
 
plt.setp( 
    axes[1].get_xticklabels(), 
    rotation=15, 
    ha='right' 
) 
 
plt.tight_layout() 
 
plt.savefig( 
    'imgs/a10_ensemble.png', 
    dpi=130 
) 
 
print( 
"\nObservation: Both ensemble methods " 
"(Random Forest, Gradient Boosting) outperform the" 
) 
print( 
"single Decision Tree baseline by reducing variance " 
"(bagging) / bias (boosting) respectively." 
)