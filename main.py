import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,  MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import main_imputer as imp

# def knn_train_predict(target_cols,k,data):
    
#     for target_col in target_cols:
#         le_y = LabelEncoder()
#         y = le_y.fit_transform(y)

#         # Split and train
#         X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)   
#         knn = KNeighborsClassifier(n_neighbors=k)
#         knn.fit(X_train, y_train)

#         # Evaluate
#         y_test_pred = knn.predict(X_test)
#         acc_score = accuracy_score(y_test, y_test_pred)
#         print(f"Model training complete. Accuracy on test set: {acc_score}")

        
#         if mask.any():
#             # Prepare features for prediction
#             X_pred = updated_data[feature_columns].copy()
            
#             # Apply same transformations using stored encoders
#             for column in label_cols:
#                 X_pred[column] = le_dict[column].transform(X_pred[column])
            
#             X_pred[numerical_cols] = scaler.transform(X_pred[numerical_cols])
#             X_pred_encoded = ct.transform(X_pred[mask])
            
#             # Predict and transform back
#             predictions = knn.predict(X_pred_encoded)
#             text_predictions = le_y.inverse_transform(predictions)
#             updated_data.loc[mask, target_column] = text_predictions

    
#     return knn





# Define specific feature sets for each target
non_imputable = ['age', 'educational-num', 'race', 'gender', 'hours-per-week', 'income']
# imputable_columns=["workclass","occupation","capital-gain","capital-loss"]
imputable_columns=["workclass"]
numeric_cols = ['age', 'educational-num', 'hours-per-week']
feature_selected_out = ['native-country','fnlwgt','education','marital-status','relationship']

data = pd.read_csv('adult.csv')
# print(data.head())
data = imp.cleaning_features(data,numeric_cols,feature_selected_out)
imp.adult_imputer(imputable_columns,5,data)
# knn_train_predict(imputable_columns,5,data)

# 
    



