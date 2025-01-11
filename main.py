import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import main_imputer as imp
import artificial_nn as ann
from sklearn.preprocessing import LabelEncoder,  MinMaxScaler, StandardScaler


def train_knn(target_col, k, data,scaler):
    print("inside train_knn")
    le = LabelEncoder()
    
    
    #save nan values 
    missing_data = data[data.isna().any(axis=1)]
    
    #from the missing data drop the target columns aswell
    Missing_data_2 = missing_data.drop(columns=target_col, axis=1)
    
    #save non nan values 
    complete_data = data.dropna()
    
    #from the complete data drop the target columns aswell
    reduced_data = complete_data.drop(columns=target_col, axis=1)
    imputer_testing_set=complete_data.copy()
    for n in target_col:
        x = reduced_data
        y = complete_data[n]
        
        # Use KNeighborsRegressor for continuous variables and KNeighborsClassifier for categorical ones
        
        y = le.fit_transform(y)
        knn = KNeighborsClassifier(n_neighbors=k)
        

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
        knn.fit(x_train, y_train)
        
        # Test accuracy
        y_pred = knn.predict(x_test)
        y_prediction=le.inverse_transform(y_pred)
        correct_prediction=le.inverse_transform(y_test)
        

        print(f"\nAccuracy score for {n}: {accuracy_score(y_test, y_pred)}")
        
        ####################################################################
        indices_in_x_test = x_test.index
        imputer_testing_set.loc[indices_in_x_test, target_col] = np.nan
        

        imputer_data=imp.adult_imputer_dt(target_col,10,imputer_testing_set,scaler)
           
        classifier_matches = 0
        imputer_matches = 0

        # Loop through indices OF X_TEST AND COMPARE WHAT THE KNN CLASSIFIER PREDICTED WITH THE CORRECT_PREDICTION AND THE IMPUTER
        for idx in indices_in_x_test:
            # Get the position of the current index in x_test
            idx_position = indices_in_x_test.get_loc(idx)
            true_value = correct_prediction[idx_position]  # Access numpy array directly
            classifier_prediction = y_prediction[idx_position]  # Classifier prediction for the index
            imputed_value = imputer_data.loc[idx, n]  # Imputed value for column `n` and index `idx`

            # Compare classifier prediction to the true value
            if classifier_prediction == true_value:
                classifier_matches += 1

            # Compare imputed value to the true value
            if imputed_value == true_value:
                imputer_matches += 1

        # Calculate accuracy for both classifier and imputer
        classifier_accuracy = (classifier_matches / len(indices_in_x_test)) * 100
        imputer_accuracy = (imputer_matches / len(indices_in_x_test)) * 100

        # Print results
        print(f"\nClassifier Accuracy for {n}: {classifier_accuracy:.2f}%")
        print(f"Imputer Accuracy for {n}: {imputer_accuracy:.2f}%")


        ####################################################################
        """ predict_missing_target = knn.predict(Missing_data_2)

        imputed_target = le.inverse_transform(np.round(predict_missing_target).astype(int))
        missing_data.loc[:, n] = imputed_target"""
        
    
   
    return missing_data






def correlation_matrix(data):
    correlation_matrix = data.corr(method='pearson')
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix Heatmap of Adult Income Dataset')
    plt.show()




# Define specific feature sets for each target
non_imputable = ['age', 'educational-num', 'race', 'gender', 'hours-per-week', 'income','marital-status','relationship']
# imputable_columns=["workclass","occupation","capital-gain","capital-loss"]
imputable_columns=['workclass','occupation','capital-gain']
imputable_columns_knn_clasifier=['workclass','occupation']
numeric_cols = ['age', 'educational-num', 'hours-per-week']
feature_selected_out = ['fnlwgt','education']




data = pd.read_csv('adult.csv')

#call the function the the file main_imputer to celan the data 
data,encoder,scaler = imp.cleaning_features(data,numeric_cols,feature_selected_out)
ann.ann_main(data)

