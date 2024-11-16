import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
file_path = 'adult.csv'
data = pd.read_csv(file_path)

####################################################################################################################
#FIND THE ROWS THAT HAVE ? AND 99999 AND REMOVE THOSE ROWS 
######################################################################################################################
# Identify and save rows with '?' or 99999 before replacement
problematic_rows = data[(data == '?') | (data == 99999)].any(axis=1)
rows_to_save = data[problematic_rows]
rows_to_save.to_csv('problematic_rows.csv', index=False)
print(len(rows_to_save))

# Print summary of saved data
print(f"\nSaved {len(rows_to_save)} rows with '?' or 99999 to 'problematic_rows.csv'")

# Create clean dataset by removing problematic
clean_data = data[~problematic_rows].copy()

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



#########################################################################
#TRAIN THE KNN FOR WORKCLASS PREDICTION
#########################################################################
# Load the clean dataset
clean_data = pd.read_csv('clean_data.csv')

# Remove specified columns and set up features/target
columns_to_drop = ['capital-gain', 'native-country', 'occupation']
X = clean_data.drop(columns=columns_to_drop + ['workclass'])  # Drop specified columns + target
y = clean_data['workclass']  # Set target

# Identify numeric and categorical columns in X
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Label encode categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # Store encoder for later use
    # Print mapping for reference
    unique_mappings = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"\n{col} encoding mapping:")
    print(unique_mappings)

print("\nEncoded feature matrix shape:", X.shape)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,  # for reproducibility
    stratify=y  # maintain class distribution
)

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
print("\nTraining KNN model...")
knn.fit(X_train, y_train)

# Make predictions on test set
y_test_pred = knn.predict(X_test)

# Print test results
print("\nTest Results:")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))
#########################################################################
######################################################################




#########################################################################
#PREDICT WORKCLASS FOR PROBLEMATIC ROWS
#########################################################################
# Load problematic rows
problem_data = pd.read_csv('problematic_rows.csv')

# Create X for problematic data by dropping unnecessary columns
problem_X = problem_data.drop(columns=columns_to_drop + ['workclass'])  # Drop same columns as training

# Label encode categorical columns in problem_X
for col in categorical_columns:
    le = LabelEncoder()
    problem_X[col] = le.fit_transform(problem_X[col])

print("\nEncoded feature matrix shape:", problem_X.shape)

# Make predictions using the trained KNN model
predicted_workclass = knn.predict(problem_X)

# Print sample of predictions
print("\nSample of predictions:")
print(predicted_workclass[:5])
print(f"\nTotal predictions made: {len(predicted_workclass)}")

# Create new dataframe with predictions replacing '?' values
final_predictions = problem_data.copy()
mask = final_predictions['workclass'] == '?'  # Find rows where workclass is '?'
final_predictions.loc[mask, 'workclass'] = predicted_workclass[mask]

# Save the updated data
final_predictions.to_csv('problematic_rows_with_predictions.csv', index=False)

# Print verification
print("\nNumber of '?' values remaining:", (final_predictions['workclass'] == '?').sum())
print("\nSample of updated workclass values:")
print(final_predictions['workclass'].head())

###########################################################################################


###########################################################################################
#TRAIN THE KNN FOR OCCUPATION PREDICTION
########################################################################################### 
# Prepare features and target for occupation prediction
X_occupation = clean_data.drop(columns=['occupation', 'native-country', 'capital-gain'])
y_occupation = clean_data['occupation']

# Identify numeric and categorical columns
numeric_columns_occ = X_occupation.select_dtypes(include=['int64', 'float64']).columns
categorical_columns_occ = X_occupation.select_dtypes(include=['object']).columns

# Scale numeric features
scaler_occ = StandardScaler()
X_occupation[numeric_columns_occ] = scaler_occ.fit_transform(X_occupation[numeric_columns_occ])

# Label encode categorical features
label_encoders_occ = {}
for col in categorical_columns_occ:
    le = LabelEncoder()
    X_occupation[col] = le.fit_transform(X_occupation[col])
    label_encoders_occ[col] = le

# Split data for occupation prediction
X_train_occ, X_test_occ, y_train_occ, y_test_occ = train_test_split(
    X_occupation, y_occupation,
    test_size=0.2,
    random_state=42,
    stratify=y_occupation
)

# Train KNN model for occupation
knn_occupation = KNeighborsClassifier(n_neighbors=5)
knn_occupation.fit(X_train_occ, y_train_occ)

# Evaluate model
y_test_pred_occ = knn_occupation.predict(X_test_occ)
print("\nOccupation Model Performance:")
print("Test Accuracy:", accuracy_score(y_test_occ, y_test_pred_occ))
print("\nClassification Report:")
print(classification_report(y_test_occ, y_test_pred_occ))
###########################################################################################


###########################################################################################
#PREDICT OCCUPATION FOR PROBLEMATIC ROWS
###########################################################################################
# Load the problematic rows with predicted workclass
problem_data_updated = pd.read_csv('problematic_rows_with_predictions.csv')

# First, handle the 'Never-worked' case by setting occupation to None
never_worked_mask = problem_data_updated['workclass'] == 'Never-worked'
problem_data_updated.loc[never_worked_mask, 'occupation'] = None

# Prepare features for occupation prediction (use same columns as training)
problem_X_occ = problem_data_updated.drop(columns=['occupation', 'native-country', 'capital-gain'])

# Only predict for rows that aren't 'Never-worked'
prediction_mask = ~never_worked_mask
rows_to_predict = problem_X_occ[prediction_mask].copy()

# Scale numeric features using the same scaler
rows_to_predict[numeric_columns_occ] = scaler_occ.transform(rows_to_predict[numeric_columns_occ])

# Label encode categorical features using the same encoders
for col in categorical_columns_occ:
    rows_to_predict[col] = label_encoders_occ[col].transform(rows_to_predict[col])

# Make predictions only for relevant rows
predictions = knn_occupation.predict(rows_to_predict)

# Update the original dataframe with predictions (only for non-Never-worked rows)
problem_data_updated.loc[prediction_mask, 'occupation'] = predictions

# Save the final predictions
problem_data_updated.to_csv('final_predictions.csv', index=False)

# Print summary
print("\nFinal Summary:")
print(f"Total rows processed: {len(problem_data_updated)}")
print(f"Rows with Never-worked (skipped): {never_worked_mask.sum()}")
print(f"Rows with predictions: {prediction_mask.sum()}")

###########################################################################################

