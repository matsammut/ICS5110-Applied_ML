import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer

def cleaning_features(data, numeric_cols,drop_columns):
    le = LabelEncoder()
    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False)

    data.replace({'?': np.nan, 99999: np.nan}, inplace=True)
    #check which columns have missing values
    
    # 1. Scale numerical features
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    # 2. Label encode gender and income
    data['gender'] = le.fit_transform(data['gender'])
    data['income'] = le.fit_transform(data['income'])
    
    # 3. One-hot encode race
    
    race_encoded = encoder.fit_transform(data[['race']])
    race_encoded_cols = encoder.get_feature_names_out(['race'])
    race_encoded_df = pd.DataFrame(race_encoded, columns=race_encoded_cols, index=data.index)
    # Combine the encoded data with original dataframe
    data = pd.concat([data.drop('race', axis=1), race_encoded_df], axis=1)
    # Binarize native country
    data['native-country'] = data['native-country'].apply(lambda x: x == 'United-States')
    data['native-country'] = data['native-country'].astype(int)

    print(data.head(10))

    data = data.drop(columns=drop_columns, axis=1)

    return data, encoder, scaler


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
        
        # Update the original dataframe
        #data[target_col] = imputed_target
        data[target_col] = imputed_values

    data.to_csv('imputed_dataset.csv', index=False)
    return data


def adult_imputer_dt(target_cols, k, data,scaler_2):
    le = LabelEncoder()
    scaler = StandardScaler()  # Initialize the scaler
    data_impute = data.drop(columns=target_cols, axis=1)

    for target_col in target_cols:
        print(f"\nProcessing {target_col}...")
        
        if target_col == 'capital-gain':
            # For capital-gain, use numerical imputation directly
            data_impute[target_col] = data[target_col]  # Retain original values for capital-gain
            
            # Apply standard scaling
            data_impute[target_col] = scaler.fit_transform(data_impute[[target_col]])  # Scale the capital-gain
            
            # Apply KNN imputation
            imputer = KNNImputer(n_neighbors=k)
            data_with_target = pd.DataFrame(imputer.fit_transform(data_impute), 
                                          columns=data_impute.columns)
            
            # Inverse scale the imputed values for capital-gain
            imputed_values = data_with_target[target_col].values
            imputed_values = scaler.inverse_transform(imputed_values.reshape(-1, 1)).flatten()  # Inverse scale
            
            # No need for label encoding/decoding for numerical values
            data[target_col] = imputed_values
            print(data.isna().sum())

        else:
            # Original code for categorical columns
            target_data = data[target_col].copy()
            target_encoded = le.fit_transform(target_data[target_data.notna()])
            target_data[target_data.notna()] = target_encoded
            data_impute[target_col] = target_data
            print(data_impute.head(10))
            # Apply KNN imputation
            imputer = KNNImputer(n_neighbors=k)
            data_with_target = pd.DataFrame(imputer.fit_transform(data_impute), 
                                          columns=data_impute.columns)
            
            imputed_values = data_with_target[target_col].values
            imputed_target = le.inverse_transform(np.round(imputed_values).astype(int))
            

            data[target_col] = imputed_target
            print(data.isna().sum())

    data[['age', 'educational-num', 'hours-per-week']] = np.round(scaler_2.inverse_transform(data[['age', 'educational-num', 'hours-per-week']]))    
    data.to_csv('imputed_dataset_2.csv', index=False)
    return data
