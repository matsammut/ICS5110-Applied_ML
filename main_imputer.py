import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

# Load the dataset
file_path = 'adult.csv'
data = pd.read_csv(file_path)

# Check for '?' values in each column
question_marks = (data == '?').sum()
print("\nColumns with '?' values:")
print(question_marks[question_marks > 0])

# Check for '99999' values in each column
missing_99999 = (data == 99999).sum()
print("\nColumns with '99999' values:")
print(missing_99999[missing_99999 > 0])

# Replace both '?' and 99999 with np.nan
data.replace(["?", 99999], np.nan, inplace=True)
data = data.replace('nan', np.nan)

def impute_column(data, target_col, feature_cols, n_neighbors=5):
    print(f"\nImputing {target_col}")
    print(feature_cols)
    print(f"Missing values before imputation: {data[target_col].isnull().sum()}")
    
    # Create features for KNN imputation
    features_for_imputation = data[feature_cols].copy()
    
    # Encode all categorical columns in features
    encoders = {}
    for col in features_for_imputation.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        non_null_values = features_for_imputation[col].dropna()
        le.fit(non_null_values)
        features_for_imputation.loc[non_null_values.index, col] = le.transform(non_null_values)
        encoders[col] = le
    
    # Encode target column
    target_le = LabelEncoder()
    non_null_values = data[target_col].dropna()
    target_le.fit(non_null_values)
    target_numeric = pd.Series(index=data.index)
    target_numeric[non_null_values.index] = target_le.transform(non_null_values)
    features_for_imputation[target_col] = target_numeric
    
    # Initialize and apply KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_values = imputer.fit_transform(features_for_imputation)
    
    # Extract imputed values and convert back to original categories
    imputed_target = pd.Series(imputed_values[:, -1])
    data[target_col] = target_le.inverse_transform(imputed_target.astype(int))
    
    print(f"Missing values after imputation: {data[target_col].isnull().sum()}")
    return data

# Impute workclass using more demographic and education features
data = impute_column(data, 
                    'workclass', 
                    ['age', 'educational-num', 'education',  'relationship', 'race', 'gender','income'])

# Impute occupation using work-related and demographic features
data = impute_column(data, 
                    'occupation', 
                    ['age',  'educational-num', 'education', 'workclass', 'marital-status', 'relationship', 'race', 'gender','income'])

# Impute native-country using demographic and socioeconomic features
data = impute_column(data, 
                    'native-country', 
                    ['age', 'educational-num', 'education', 'workclass', 'occupation', 'race', 'gender','income'])

# Impute capital-gain using work and demographic features
data = impute_column(data, 
                    'capital-gain', 
                    ['age', 'educational-num', 'education', 'workclass', 'occupation', 'marital-status', 'relationship', 'race', 'gender','income' ])

# Save the results
data.to_csv('imputed_dataset.csv', index=False)
print("\nFirst few rows of final imputed data:")
print(data[['age', 'workclass', 'occupation', 'native-country', 'capital-gain']].head())
