import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
import pickle

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
   # data['workclass'] = le.fit_transform(data['workclass'])

    with open('label_encoder_work.pkl', 'wb') as le_file:
        pickle.dump(le, le_file)

    #data['occupation'] = le.fit_transform(data['occupation'])
    with open('label_encoder_occ.pkl', 'wb') as le_file:
        pickle.dump(le, le_file)
    
    #columns_to_encode = ['race','marital-status','relationship']
    columns_to_encode = ['race']
    # 3. One-hot encode race
    for N in columns_to_encode:
        race_encoded = encoder.fit_transform(data[[N]])
        race_encoded_cols = encoder.get_feature_names_out([N])
        race_encoded_df = pd.DataFrame(race_encoded, columns=race_encoded_cols, index=data.index)
        # Combine the encoded data with original dataframe
        data = pd.concat([data.drop(N, axis=1), race_encoded_df], axis=1)
    # Binarize native country
    data['native-country'] = data['native-country'].apply(lambda x: x == 'United-States')
    data['native-country'] = data['native-country'].astype(int)

    

    data = data.drop(columns=drop_columns, axis=1)

    
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    with open('race_onehot_encoder.pkl', 'wb') as enc_file:
        pickle.dump(encoder, enc_file)

    return data, encoder, scaler

def adult_imputer(target_cols, k, data, numeric_cols):
    le = LabelEncoder()
    data_impute = data.drop(columns=target_cols, axis=1)

    for target_col in target_cols:
        print(f"\nProcessing {target_col}...")
        
        if target_col in numeric_cols:
            # Store original values without scaling for imputation
            data_impute[target_col] = data[target_col]
            
            # Apply KNN imputation on raw values
            imputer = KNNImputer(n_neighbors=k)
            data_with_target = pd.DataFrame(imputer.fit_transform(data_impute), columns=data_impute.columns)
            
            # After imputation, scale all numeric columns
            
            
        else:
            # Original categorical imputation logic
            target_data = data[target_col].copy()
            target_encoded = le.fit_transform(target_data[target_data.notna()])
            target_data[target_data.notna()] = target_encoded
            data_impute[target_col] = target_data
            
            imputer = KNNImputer(n_neighbors=k)
            data_with_target = pd.DataFrame(imputer.fit_transform(data_impute), columns=data_impute.columns)
            imputed_values = data_with_target[target_col].values
            data[target_col] = imputed_values
    
    
    data.to_csv('imputed_dataset.csv', index=False)
    return data



    print(data.isna().sum())
    data=data.dropna()
    
    data[['age', 'educational-num', 'hours-per-week']] = np.round(scaler.inverse_transform(data[['age', 'educational-num', 'hours-per-week']]))   
    print(data.isna().sum())
    data.to_csv('Decision_tree_datasets/orignal_data_droped_nan_values.csv', index=False)