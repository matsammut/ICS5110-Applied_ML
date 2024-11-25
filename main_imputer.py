import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer

def cleaning_features(data, numeric_cols,drop_columns):
    le = LabelEncoder()
    scaler = StandardScaler()

    data.replace({'?': np.nan, 99999: np.nan}, inplace=True)

    # 1. Scale numerical features
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    # 2. Label encode gender and income
    data['gender'] = le.fit_transform(data['gender'])
    data['income'] = le.fit_transform(data['income'])

    # 3. One-hot encode race
    race_encoded = pd.get_dummies(data['race'], prefix='race')
    data = pd.concat([data.drop('race', axis=1), race_encoded], axis=1)

    data = data.drop(columns=drop_columns, axis=1)

    return data


def adult_imputer(target_cols,k,data):
    le = LabelEncoder()

    data_impute = data.drop(columns=target_cols, axis=1)

    for target_col in target_cols:
        print(f"\nProcessing {target_col}...")
        
        # Label encode the target column first
        target_data = data[target_col].copy()
        target_encoded = le.fit_transform(target_data[target_data.notna()]) # Encode non-null values
        target_data[target_data.notna()] = target_encoded
        data_impute[target_col] = target_data
        

        #change from label encoding to one hot for target as it doesnt make sense with catagories
        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=k)
        data_with_target = pd.DataFrame(imputer.fit_transform(data_impute), columns=data_impute.columns)
        imputed_values = data_with_target[target_col].values
        #imputed_target = le.inverse_transform(np.round(imputed_values).astype(int))
        #print(imputed_target)
        #imputed_target.to_csv('imputed_dataset.csv', index=False)
        
        # Update the original dataframe
        data[target_col] = imputed_values

    data.to_csv('imputed_dataset.csv', index=False)
