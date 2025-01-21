import pandas as pd
import main_imputer as imp





# Define specific feature sets for each target
non_imputable = ['age', 'educational-num', 'race', 'gender', 'hours-per-week', 'income']
# imputable_columns=["workclass","occupation","capital-gain","capital-loss"]
imputable_columns=['workclass','occupation','capital-gain']

numeric_cols = ['age', 'educational-num', 'hours-per-week']
feature_selected_out = ['fnlwgt','education','marital-status','relationship']




data = pd.read_csv('adult.csv')

#call the function the the file main_imputer to celan the data 
data,encoder,scaler = imp.cleaning_features(data,numeric_cols,feature_selected_out)


print("\n---------------------------------------------")
print(f"replace missing data with imputer ")
print("---------------------------------------------\n")
imputer_data=imp.adult_imputer(imputable_columns, 5, data, numeric_cols)




