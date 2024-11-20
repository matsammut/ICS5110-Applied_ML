import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

def cleaned_data_function(file_path):
    data = pd.read_csv(file_path)

    # Identify and save rows with '?' or 99999 before replacement
    problematic_rows = data[(data == '?') | (data == 99999)].any(axis=1)
    rows_to_save = data[problematic_rows]
    rows_to_save.to_csv('problematic_rows.csv', index=False)
    print(len(rows_to_save))

    # Print summary of saved data
    # Create clean dataset by removing problematic rows
    clean_data = data[~problematic_rows].copy()

    # Print summary of the clean dataset
    print(f"\nOriginal dataset size: {len(data)}")
    print(f"Rows removed: {len(rows_to_save)}")
    print(f"Clean dataset size: {len(clean_data)}")

    # Save clean dataset
    clean_data.to_csv('clean_data.csv', index=False)
    

    # Verify no '?' or 99999 remain in clean_data
    remaining_questions = (clean_data == '?').sum().sum()
    remaining_nines = (clean_data == 99999).sum().sum()
    print(f"\nVerification - remaining problematic values: {remaining_questions + remaining_nines}")


def knn_workclass_train(feature_columns,target_column):
    
    clean_data = pd.read_csv('clean_data.csv')
    
    # SPLIT THE FEATURES AND THE TARGET VALUES 
    X = clean_data[feature_columns].copy()
    y = clean_data[target_column]

    ###################################################################################################                                                     
    column_names = X.columns
    print(f"The columns to be used are {column_names}")

    # Label encode gender and income
    label_cols = ['gender', 'income']
    for column in label_cols:
        le_x = LabelEncoder()
        X[column] = le_x.fit_transform(X[column])

    # Scale numerical columns
    numerical_cols = ['age', 'educational-num', 'hours-per-week']
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # One-hot encode education and race
    onehot_cols = ['race', 'education']
    ct = ColumnTransformer([('onehot', OneHotEncoder(drop='first', sparse_output=False), onehot_cols)], remainder='passthrough')
    
    # Transform the data and create new dataframe with proper column names
    X_encoded = ct.fit_transform(X)
    
    # Get the new column names after one-hot encoding
    onehot_feature_names = ct.named_transformers_['onehot'].get_feature_names_out(onehot_cols)
    # Get the names of the columns that weren't transformed
    passthrough_cols = [col for col in X.columns if col not in onehot_cols]
    # Combine all column names
    new_column_names = list(onehot_feature_names) + passthrough_cols
    
    # Convert to DataFrame with proper column names
    X_encoded_df = pd.DataFrame(X_encoded, columns=new_column_names)
    
    # Label encode the target
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)

    #################################################################################################
    #split the data into training and testing 
    X_train, X_test, y_train, y_test = train_test_split(X_encoded_df, y, test_size=0.1, random_state=42)   

    #################################################################################################
    #train the KNN model 
    knn = KNeighborsClassifier(n_neighbors=100)
    knn.fit(X_train, y_train)   

    #evaluate the model 
    y_test_pred = knn.predict(X_test)
    acc_score=accuracy_score(y_test, y_test_pred)
    print(f"Accuracy: {acc_score}")
    
    return knn  # Just return the trained model

def prediction_function(knn, target_column='workclass', feature_columns=['age','education','educational-num', 'race', 'gender', 'hours-per-week', 'income']):
    # Try to read the running updated file, if it doesn't exist, use the original problematic rows
    try:
        updated_data = pd.read_csv('problematic_rows_running_updates.csv')
    except FileNotFoundError:
        updated_data = pd.read_csv('problematic_rows.csv')
    
    # Create X for the problematic rows
    X = updated_data[feature_columns].copy()
    
    # Create label encoders dictionary
    label_encoders = {}
    
    # Instead of one-hot encoding, use the same label encoding as in training
    for column in ['education', 'race', 'gender', 'income']:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])
    
    # Get the target column from the model
    target = target_column
    
    print(f"\nProcessing {target}...")
    
    # Create label encoder for target column and fit it on the clean data
    clean_data = pd.read_csv('clean_data.csv')
    target_encoder = LabelEncoder()
    target_encoder.fit(clean_data[target])
    
    # Use the passed model directly
    mask = updated_data[target] == '?'
    if mask.any():
        predictions = knn.predict(X[mask])
        # Convert numeric predictions back to text labels
        text_predictions = target_encoder.inverse_transform(predictions)
        updated_data.loc[mask, target] = text_predictions
    
    print(f"Number of '?' remaining in {target}: {(updated_data[target] == '?').sum()}")
    print(f"Sample of updated {target} values:")
    print(updated_data[target].head())
    
    # Save the running updated dataset
    updated_data.to_csv('problematic_rows_running_updates.csv', index=False)
    print(f"\nUpdated data saved to 'problematic_rows_running_updates.csv'")


#column numbers from cleaned dataset
"""
0=age
1=workclass
2=fnlwgt
3=education 
4=education-num
5=marital-status
6=occupation
7=relationship
8=race
9=gender
10=capital-gain
11=capital-loss
12=hours-per-week
13=native_country
14=income 
"""

# Define specific feature sets for each target
workclass_features = ['age','education', 'educational-num', 'race', 'gender', 'hours-per-week', 'income']
occupation_features = ['age', 'education', 'educational-num', 'race', 'gender', 'hours-per-week', 'income']
native_country_features = ['age', 'education', 'educational-num', 'race', 'gender', 'hours-per-week', 'income']

# Step 1: load the adult dataset, find all the ? and 99999 and replace them with nan 
cleaned_data_function(file_path = 'adult.csv')

# Train and predict for workclass
knn_workclass = knn_workclass_train(feature_columns=workclass_features,target_column='workclass')
#prediction_function(knn_workclass, target_column='workclass', feature_columns=workclass_features)

# Train and predict for occupation
knn_occupation = knn_workclass_train(feature_columns=occupation_features,target_column='occupation')
prediction_function(knn_occupation, target_column='occupation', feature_columns=occupation_features)

# Train and predict for native-country
knn_native = knn_workclass_train(feature_columns=native_country_features,target_column='native-country')
prediction_function(knn_native, target_column='native-country', feature_columns=native_country_features)

# Load both CSV files and compare results
imputed_data = pd.read_csv('imputed_dataset.csv')
final_data = pd.read_csv('problematic_rows_running_updates.csv')

# Columns to compare
columns_to_compare = ['workclass', 'occupation', 'native-country']

# Calculate similarities
for column in columns_to_compare:
    matches = (final_data[column] == imputed_data.loc[final_data.index, column]).sum()
    total = len(final_data)
    similarity = (matches / total) * 100
    
    print(f"\nSimilarity for {column}:")
    print(f"Matching values: {matches} out of {total}")
    print(f"Similarity percentage: {similarity:.2f}%")

# Overall similarity
total_matches = sum(final_data[col] == imputed_data.loc[final_data.index, col] for col in columns_to_compare).sum()
total_values = len(final_data) * len(columns_to_compare)
overall_similarity = (total_matches / total_values) * 100

print(f"\nOverall similarity across all columns: {overall_similarity:.2f}%")