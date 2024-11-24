import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
df = pd.read_csv('adult.csv')

# Replace missing values with NaN
df.replace({'?': np.nan, 99999: np.nan}, inplace=True)

# Define columns
feature_cols = ['age', 'educational-num', 'race', 'gender', 'hours-per-week', 'income']
target_cols = ['workclass', 'occupation', 'native-country']

# Create dummy variables for categorical features
X = pd.get_dummies(df[feature_cols], columns=['race', 'gender', 'income'])

# Impute each target column
for target_col in target_cols:
    print(f"\nProcessing {target_col}...")
    
    # Split into rows with and without missing values
    train_mask = df[target_col].notna()
    missing_mask = df[target_col].isna()
    
    if missing_mask.any():
        # Train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X[train_mask], df[target_col][train_mask])
        
        # Predict missing values
        predictions = knn.predict(X[missing_mask])
        df.loc[missing_mask, target_col] = predictions
        
        print(f"Imputed {missing_mask.sum()} missing values in {target_col}")

# Save results
df.to_csv('imputed_dataset.csv', index=False)
print("\nImputation complete. Results saved to 'imputed_dataset.csv'")
