import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

def cleaned_data_function(file_path):
    data = pd.read_csv(file_path)

    # Identify and save rows with '?' or 99999 before replacement
    problematic_rows = data[(data == '?') | (data == 99999)].any(axis=1)
    rows_to_save = data[problematic_rows]
    rows_to_save.to_csv('Missing_data_rows.csv', index=False)
    print(len(rows_to_save))

    # Print summary of saved data
    # Create clean dataset by removing problematic rows
    clean_data = data[~problematic_rows].copy()

    # Save clean dataset
    clean_data.to_csv('clean_data.csv', index=False)#
    return problematic_rows
    
def knn_workclass_train_predict(feature_columns, target_column):
    """
    Trains a KNN model on the clean dataset and uses it to predict missing values for the 'workclass' column.
    """
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
    onehot_cols = ['race']
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
    # Split the data into training and testing 
    X_train, X_test, y_train, y_test = train_test_split(X_encoded_df, y, test_size=0.1, random_state=42)   

    # Train the KNN model 
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)  # Fit the model to the training data

    # Evaluate the model 
    y_test_pred = knn.predict(X_test)
    acc_score = accuracy_score(y_test, y_test_pred)
    print(f"Model training complete. Accuracy on test set: {acc_score}")

    #################################################################################################

    # PREDICT MISSING DATA
    try:
        updated_data = pd.read_csv('problematic_rows_running_updates.csv')
    except FileNotFoundError:
        updated_data = pd.read_csv('Missing_data_rows.csv')
    
    # Create X for the problematic rows
    X = updated_data[feature_columns].copy()

    # Label encode gender and income
    for column in label_cols:
        le_x = LabelEncoder()
        X[column] = le_x.fit_transform(X[column])

    # Scale numerical columns
    X[numerical_cols] = scaler.transform(X[numerical_cols])

    # One-hot encode education and race
    X_encoded = ct.transform(X)
    
    # Convert to DataFrame with proper column names
    X_encoded_df = pd.DataFrame(X_encoded, columns=new_column_names)
    
    # Label encode the target column for prediction
    target_encoder = LabelEncoder()
    target_encoder.fit(clean_data[target_column])
    
    mask = (updated_data[target_column] == '?') | (updated_data[target_column] == 99999)
    
    if mask.any():
        predictions = knn.predict(X_encoded_df[mask])
        text_predictions = target_encoder.inverse_transform(predictions)
        updated_data.loc[mask, target_column] = text_predictions

    print(f"Number of '?' or '99999' remaining in '{target_column}': {((updated_data[target_column] == '?') | (updated_data[target_column] == 99999)).sum()}")

    # Save the updated data
    updated_data.to_csv('problematic_rows_running_updates.csv', index=False)
    print(f"Updated data saved to 'problematic_rows_running_updates.csv'")
    
    return knn  # Return the trained model for potential further use

def compare_imputedknn_vs_knn():
        
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

        # After your existing commented comparison code, add:
        # Merge clean and problematic data
        print("\nMerging clean and imputed datasets...")
        clean_data = pd.read_csv('clean_data.csv')
        problematic_data = pd.read_csv('problematic_rows_running_updates.csv')




# Define specific feature sets for each target
workclass_features = ['age', 'educational-num', 'race', 'gender', 'hours-per-week', 'income']
occupation_features = ['age', 'educational-num', 'race', 'gender', 'hours-per-week', 'income']
native_country_features = ['age', 'educational-num', 'race', 'gender', 'hours-per-week', 'income']
capital_gain_features = ['age', 'educational-num', 'race', 'gender', 'hours-per-week', 'income']

# Step 1: load the adult dataset, find all the ? and 99999 and replace them with nan 
cleaned_data_function(file_path = 'adult.csv')



target_predicitons=["workclass","occupation","native-country","capital-gain"]
for class_predict in target_predicitons:
    knn_workclass_train_predict(workclass_features,class_predict)

compare_imputedknn_vs_knn()
#


"""
# Combine the datasets
complete_dataset = pd.concat([clean_data, problematic_data], axis=0, ignore_index=True)

# Sort by index to maintain original order (optional)
complete_dataset = complete_dataset.sort_index()

# Save the complete dataset
complete_dataset.to_csv('complete_dataset.csv', index=False)
print(f"Complete dataset saved with {len(complete_dataset)} rows")
print(f"- Clean data rows: {len(clean_data)}")
print(f"- Imputed rows: {len(problematic_data)}")"""