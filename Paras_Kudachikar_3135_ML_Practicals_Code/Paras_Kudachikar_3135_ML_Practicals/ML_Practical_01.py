# Machine Learning Laboratory - Practical No. 01
# Name: Paras Kudachikar | Roll No.: 3135

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
# Sample open-access style dataset (HR/Employee attrition style)
data = {
    'Age': [25, 32, np.nan, 45, 29, 38, np.nan, 51],
    'Department': ['Sales', 'IT', 'HR', 'IT', 'Sales', np.nan, 'HR', 'IT'],
    'Salary': [40000, 55000, 48000, np.nan, 42000, 60000, 47000, 72000],
    'YearsExperience': [1, 5, 3, 12, 2, 8, 4, 15],
    'Attrition': ['Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'No']}
df = pd.DataFrame(data)
print("Original Dataset:\n", df, "\n")
print("Missing values per column:\n", df.isnull().sum(), "\n")
# Step 1: Handle missing values
num_imputer = SimpleImputer(strategy='mean')
df['Age'] = num_imputer.fit_transform(df[['Age']])
df['Salary'] = num_imputer.fit_transform(df[['Salary']])
cat_imputer = SimpleImputer(strategy='most_frequent')
df['Department'] = cat_imputer.fit_transform(df[['Department']]).ravel()
print("After Handling Missing Values:\n", df, "\n")

# Step 2: Encode categorical variables
le = LabelEncoder()
df['Attrition_Encoded'] = le.fit_transform(df['Attrition'])
df_encoded = pd.get_dummies(df, columns=['Department'], prefix='Dept')
print("After One-Hot Encoding 'Department':\n", df_encoded, "\n")
# Step 3: Scale numerical attributes
scaler = StandardScaler()
num_cols = ['Age', 'Salary', 'YearsExperience']
df_encoded[num_cols] = scaler.fit_transform(
 df_encoded[num_cols])
print("After Feature Scaling (Standardization):\n",
 df_encoded[num_cols].round(3),"\n")
print("Final Model-Ready Dataset Shape:", df_encoded.shape)
