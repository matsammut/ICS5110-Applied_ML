import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer

# Load the dataset
df = pd.read_csv('adult.csv')

# Replace missing values with NaN
df.replace({'?': np.nan, 99999: np.nan}, inplace=True)

# Define columns
feature_cols = ['age', 'educational-num', 'race', 'gender', 'hours-per-week', 'income']
target_cols = ['workclass', 'occupation', 'native-country','capital-gain']

# Prepare features
X = df[feature_cols].copy()

# 1. Scale numerical features
scaler = StandardScaler()
numeric_cols = ['age', 'educational-num', 'hours-per-week']
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# 2. Label encode gender and income
le = LabelEncoder()
X['gender'] = le.fit_transform(X['gender'])
X['income'] = le.fit_transform(X['income'])

# 3. One-hot encode race
race_encoded = pd.get_dummies(X['race'], prefix='race')
X = pd.concat([X.drop('race', axis=1), race_encoded], axis=1)

# Impute each target column
for target_col in target_cols:
    print(f"\nProcessing {target_col}...")
    
    # Label encode the target column first
    target_data = df[target_col].copy()
    le = LabelEncoder()
    target_encoded = le.fit_transform(target_data[target_data.notna()]) # Encode non-null values
    target_data[target_data.notna()] = target_encoded
    
    # Prepare data for imputation
    data_for_imputation = pd.concat([X, target_data], axis=1)
    
    # Apply KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    imputed_values = imputer.fit_transform(data_for_imputation)
    
    
    # Round the imputed values to nearest integer before inverse transform
    imputed_target = le.inverse_transform(np.round(imputed_values[:, -1]).astype(int))
    
    # Update the original dataframe
    df[target_col] = imputed_target
    
    print(f"Imputed missing values in {target_col}")

# Save results
df.to_csv('imputed_dataset.csv', index=False)
print("\nImputation complete. Results saved to 'imputed_dataset.csv'")


