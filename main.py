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

# Load the dataset
file_path = 'adult.csv'
data = pd.read_csv(file_path)

# Identify and save rows with '?' or 99999 before replacement
problematic_rows = data[(data == '?') | (data == 99999)].any(axis=1)
rows_to_save = data[problematic_rows]
rows_to_save.to_csv('problematic_rows.csv', index=False)
print(len(rows_to_save))

# Print summary of saved data
print(f"\nSaved {len(rows_to_save)} rows with '?' or 99999 to 'problematic_rows.csv'")

# Create clean dataset by removing problematic rows
clean_data = data[~problematic_rows].copy()

# Print summary of the clean dataset
print(f"\nOriginal dataset size: {len(data)}")
print(f"Rows removed: {len(rows_to_save)}")
print(f"Clean dataset size: {len(clean_data)}")

# Save clean dataset
clean_data.to_csv('clean_data.csv', index=False)
print("\nClean dataset saved to 'clean_data.csv'")

# Verify no '?' or 99999 remain in clean_data
remaining_questions = (clean_data == '?').sum().sum()
remaining_nines = (clean_data == 99999).sum().sum()
print(f"\nVerification - remaining problematic values: {remaining_questions + remaining_nines}")



"""

#########################################################################
#TRAIN THE KNN FOR WORKCLASS PREDICTION
#########################################################################
# Load the clean dataset
clean_data = pd.read_csv('clean_data.csv')

####
#SPLIT THE FEATURES AND THE TARGET VALUES 
X = clean_data.iloc[:, [3, 4, 8, 9, 12, 14]]
y = clean_data.iloc[:, 1]


###################################################################################################                                                     
column_names = X.columns
print(column_names)

#perform label encoding to columns 
for column in column_names[[0, 2, 3, 5]]:  # Indices in X, not in quotes
    le_x = LabelEncoder()
    X[column] = le_x.fit_transform(X[column])


#####
#perform label encoding to the target 
le_y = LabelEncoder()
y = le_y.fit_transform(y)

#################################################################################################
#split the data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

#################################################################################################
#train the KNN model 
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(X_train, y_train)   

#################################################################################################
#evaluate the model 
y_test_pred = knn.predict(X_test)
acc_score=accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {acc_score}")
#print(classification_report(y_test, y_test_pred))



#################################################################################################
#########################################################################
#TRAIN THE KNN FOR WORKCLASS PREDICTION for PROBLEMATIC ROWS workclass 
#########################################################################
problem_data = pd.read_csv('problematic_rows.csv')
#create X for the problematic rows 
problem_X = problem_data.iloc[:, [3, 4, 8, 9, 12, 14]]  

#perform label encoding to the problematic rows 
for column in column_names[[0, 2, 3, 5]]:  # Indices in X, not in quotes
    le_x = LabelEncoder()
    problem_X[column] = le_x.fit_transform(problem_X[column])

#make predictions for the problematic rows 
problem_predictions = knn.predict(problem_X)

# Replace '?' with predictions in the original problematic data
mask = problem_data['workclass'] == '?'  # Find rows where workclass is '?'
problem_data.loc[mask, 'workclass'] = le_y.inverse_transform(problem_predictions[mask])

# Save the updated problematic rows
problem_data.to_csv('problematic_rows_updated.csv', index=False)

# Verification
print("\nVerification:")
print(f"Number of '?' remaining in workclass: {(problem_data['workclass'] == '?').sum()}")
print("\nSample of updated workclass values:")
print(problem_data['workclass'].head())


#----------------------------------------------------------------------------------------------
#########################################################################
#TRAIN THE KNN FOR OCCUPATION PREDICTION
#########################################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('clean_data.csv')

# Split the features and the target
X = data.iloc[:, [1, 3, 4, 8, 9, 12, 14]]
y = data.iloc[:, 6]

# Store original column names for the selected features
column_names = X.columns
print("Original column names:", column_names)

# First, perform label encoding on all categorical passthrough columns
categorical_cols = [0, 1,2, 5]  # workclass, education, hours-per-week

# Label encode all categorical columns in passthrough
for col in [3, 4,6]:  # race, gender, and income columns
    le = LabelEncoder()
    X.iloc[:, col] = le.fit_transform(X.iloc[:, col])

# Now apply the ColumnTransformer with OneHotEncoder
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), categorical_cols)],
    remainder='passthrough'
)

# Fit and transform X
X = ct.fit_transform(X).toarray()

# Add this before the KNN training
le_y = LabelEncoder()
y = le_y.fit_transform(y)

#########################################################################################
#split the data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)   

#########################################################################################


#train the KNN model 
knn_occupation = KNeighborsClassifier(n_neighbors=10)
knn_occupation.fit(X_train, y_train)   
#########################################################################################
#evaluate the model 
y_test_pred = knn_occupation.predict(X_test)
acc_score=accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {acc_score}")
print(classification_report(y_test, y_test_pred))
#########################################################################################



#########################################################################
#predict THE KNN FOR OCCUPATION PREDICTION
#########################################################################

data = pd.read_csv('problematic_rows_updated.csv')

#create X for the problematic rows 
problem_X = data.iloc[:, [1, 3, 4, 8, 9, 12, 14]]   
print(problem_X.columns)  

# First, perform label encoding on all categorical passthrough columns
categorical_cols = [0, 1, 2, 5]  # workclass, education, hours-per-week

# Label encode all categorical columns in passthrough
for col in [3, 4, 6]:  # race, gender, and income columns
    le = LabelEncoder()
    problem_X.iloc[:, col] = le.fit_transform(problem_X.iloc[:, col])

# Now apply the same ColumnTransformer (use the one already fitted)
problem_X = ct.transform(problem_X).toarray()

# Now you can make predictions using knn_occupation
predictions = knn_occupation.predict(problem_X)
#

# Replace '?' with predictions in the original problematic data
mask = data['occupation'] == '?'  # Find rows where occupation is '?'
data.loc[mask, 'occupation'] = le_y.inverse_transform(predictions[mask])

# Save the updated problematic rows
data.to_csv('problematic_rows_occupation_update.csv', index=False)

# Verification
print("\nVerification:")
print(f"Number of '?' remaining in occupation: {(data['occupation'] == '?').sum()}")
print("\nSample of updated occupation values:")
print(data['occupation'].head())



##########################################################################################  


#########################################################################
#TRAIN THE KNN FOR native-country 
#########################################################################


#load the dataset 
data = pd.read_csv('clean_data.csv')
#########################################################################################
#get the features and the target 
X = data.iloc[:, [0,1, 3, 4,6, 8, 9, 14]]
y = data.iloc[:, 13]


# Store original column names for the selected features
column_names = X.columns
print("Original column names:", column_names)

# First, perform label encoding on all categorical passthrough columns
categorical_cols = [0, 1,2, 3,4]  # workclass, education, hours-per-week

# Label encode all categorical columns in passthrough
for col in [5, 6,7]:  # race, gender, and income columns
    le = LabelEncoder()
    X.iloc[:, col] = le.fit_transform(X.iloc[:, col])

# Now apply the ColumnTransformer with OneHotEncoder
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), categorical_cols)],
    remainder='passthrough'
)

# Fit and transform X
X = ct.fit_transform(X).toarray()

# Add this before the KNN training
le_y = LabelEncoder()
y = le_y.fit_transform(y)

#########################################################################################
#split the data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)   

#########################################################################################

#train the KNN model 
knn_native_country = KNeighborsClassifier(n_neighbors=5)
knn_native_country.fit(X_train, y_train)       

#########################################################################################
#evaluate the model 
y_test_pred = knn_native_country.predict(X_test)
acc_score=accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {acc_score}")
print(classification_report(y_test, y_test_pred))


#########################################################################
#predict THE native-country PREDICTION
#########################################################################

data_2 = pd.read_csv('problematic_rows_occupation_update.csv')    
problem_X = data_2.iloc[:, [0,1, 3, 4,6, 8, 9, 14]]   

# Label encode the categorical columns in passthrough
for col in [5, 6, 7]:  # race, gender, and income columns
    le = LabelEncoder()
    problem_X.iloc[:, col] = le.fit_transform(problem_X.iloc[:, col])

# Use the already fitted ColumnTransformer instead of creating a new one
problem_X = ct.transform(problem_X).toarray()

#########################################################################################
#prediction for native-country 
predictions = knn_native_country.predict(problem_X)

# Replace '?' with predictions in the original problematic data
mask = data_2['native-country'] == '?'  # Find rows where native-country is '?'
data_2.loc[mask, 'native-country'] = le_y.inverse_transform(predictions[mask])

# Save the updated problematic rows
data_2.to_csv('problematic_rows_final.csv', index=False)

# Verification
print("\nVerification:")
print(f"Number of '?' remaining in native-country: {(data_2['native-country'] == '?').sum()}")
print("\nSample of updated native-country values:")
print(data_2['native-country'].head())

#########################################################################################



# Load both CSV files
imputed_data = pd.read_csv('imputed_dataset.csv')
final_data = pd.read_csv('problematic_rows_final.csv')

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

"""