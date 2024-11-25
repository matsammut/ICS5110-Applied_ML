import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, Normalizer
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
    
def knn_train_predict(feature_columns, target_column):
    """
    Trains a KNN model on the clean dataset and uses it to predict missing values for the 'workclass' column.
    """
    clean_data = pd.read_csv('clean_data.csv')
    
    # Split features and target
    X = clean_data[feature_columns].copy()  
    y = clean_data[target_column]

    # Label encode gender and income
    label_cols = ['gender', 'income']
    le_dict = {}  # Store encoders for prediction
    for column in label_cols:
        le_x = LabelEncoder()
        X[column] = le_x.fit_transform(X[column])
        le_dict[column] = le_x  # Save for later use

    # Scale numerical columns
    numerical_cols = ['age', 'educational-num', 'hours-per-week']
    scaler = MinMaxScaler(feature_range=(0, 2))
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # One-hot encode education and race
    onehot_cols = ['race']
    ct = ColumnTransformer([('onehot', OneHotEncoder(drop='first', sparse_output=False), onehot_cols)], remainder='passthrough')
    X_encoded = ct.fit_transform(X)
    
    # Label encode the target
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)   
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)

    # Evaluate
    y_test_pred = knn.predict(X_test)
    acc_score = accuracy_score(y_test, y_test_pred)
    print(f"Model training complete. Accuracy on test set: {acc_score}")

    # Predict missing data
    try:
        updated_data = pd.read_csv('problematic_rows_running_updates.csv')
    except FileNotFoundError:
        updated_data = pd.read_csv('Missing_data_rows.csv')
    
    mask = (updated_data[target_column] == '?') | (updated_data[target_column] == 99999)
    
    if mask.any():
        # Prepare features for prediction
        X_pred = updated_data[feature_columns].copy()
        
        # Apply same transformations using stored encoders
        for column in label_cols:
            X_pred[column] = le_dict[column].transform(X_pred[column])
        
        X_pred[numerical_cols] = scaler.transform(X_pred[numerical_cols])
        X_pred_encoded = ct.transform(X_pred[mask])
        
        # Predict and transform back
        predictions = knn.predict(X_pred_encoded)
        text_predictions = le_y.inverse_transform(predictions)
        updated_data.loc[mask, target_column] = text_predictions

    print(f"Number of '?' or '99999' remaining in '{target_column}': {((updated_data[target_column] == '?') | (updated_data[target_column] == 99999)).sum()}")
    updated_data.to_csv('problematic_rows_running_updates.csv', index=False)
    print(f"Updated data saved to 'problematic_rows_running_updates.csv'")
    
    return knn

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
    knn_train_predict(workclass_features,class_predict)
    
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