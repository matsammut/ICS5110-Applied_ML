import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,  MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import main_imputer as imp

def train_knn(target_col, k, data):
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
    
    
    
    
    
    for n in target_col:
        x = reduced_data
        y = complete_data[n]
        print(x.head(10))
        print(y.head(10))
        y = le.fit_transform(y)
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        
        # Test accuracy
        y_pred = knn.predict(x_test)
        print(f"\nAccuracy score for {n}: {accuracy_score(y_test, y_pred)}")
        
    
        predict_missing_target = knn.predict(Missing_data_2)

        imputed_target = le.inverse_transform(np.round(predict_missing_target).astype(int))
        missing_data.loc[:, n] = imputed_target
        
    
   
    return missing_data




def compare(knn_claisifer_missing_data_inputed, imputer_data, target_col):

    indices_to_compare = knn_claisifer_missing_data_inputed.index
    
    common_indices = indices_to_compare.intersection(imputer_data.index)
    
    for n in target_col:
        workclass_df1 = knn_claisifer_missing_data_inputed.loc[common_indices, n].tolist()
        workclass_df2 = imputer_data.loc[common_indices, n].tolist()
        
        # Initialize counter for matches
        matches = 0
        
        
        # Loop through both lists simultaneously using zip
        for i in range(len(workclass_df1)):
            if workclass_df1[i] == workclass_df2[i]:
                matches += 1
        
        match_percentage = (matches / len(workclass_df1)) * 100
        print(f"Matching percentage:  for {match_percentage:.2f}%")




# Define specific feature sets for each target
non_imputable = ['age', 'educational-num', 'race', 'gender', 'hours-per-week', 'income']
# imputable_columns=["workclass","occupation","capital-gain","capital-loss"]
imputable_columns=['workclass','occupation','capital-gain']
numeric_cols = ['age', 'educational-num', 'hours-per-week']
feature_selected_out = ['native-country','fnlwgt','education','marital-status','relationship']




data = pd.read_csv('adult.csv')

#call the function the the file main_imputer to celan the data 
data = imp.cleaning_features(data,numeric_cols,feature_selected_out)
data_knn=data.copy()





imputer_data=imp.adult_imputer(imputable_columns,5,data)

#knn_claisifer_missing_data_inputed=train_knn(imputable_columns,10,data_knn)

#compare(knn_claisifer_missing_data_inputed,imputer_data,imputable_columns)

