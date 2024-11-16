import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

# Load the dataset
file_path = 'adult.csv'
data = pd.read_csv(file_path)

# Check for '?' values in each column
question_marks = (data == '?').sum()
print("\nColumns with '?' values:")
print(question_marks[question_marks > 0])

# Check for '99999' values in each column
missing_99999 = (data == 99999).sum()
print("\nColumns with '99999' values:")
print(missing_99999[missing_99999 > 0])

# Replace both '?' and 99999 with np.nan
data.replace(["?", 99999], np.nan, inplace=True)
data = data.replace('nan', np.nan)
###########################################################
### give me the avarages for the datas
column_means = data.mean(numeric_only=True)
print(column_means)
################################################################


def impute_column(data, target_col, feature_cols, n_neighbors=1000):
    print(f"\nImputing {target_col}")
    print(feature_cols)
    print(f"Missing values before imputation: {data[target_col].isnull().sum()}")
    
    # Create features for KNN imputation
    features_for_imputation = data[feature_cols].copy()
    
    # Encode all categorical columns in features
    encoders = {}
    for col in features_for_imputation.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        non_null_values = features_for_imputation[col].dropna()
        le.fit(non_null_values)
        features_for_imputation.loc[non_null_values.index, col] = le.transform(non_null_values)
        encoders[col] = le
    
    # Encode target column
    target_le = LabelEncoder()
    non_null_values = data[target_col].dropna()
    target_le.fit(non_null_values)
    target_numeric = pd.Series(index=data.index)
    target_numeric[non_null_values.index] = target_le.transform(non_null_values)
    features_for_imputation[target_col] = target_numeric
    
    # Initialize and apply KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_values = imputer.fit_transform(features_for_imputation)
    
    # Extract imputed values and convert back to original categories
    imputed_target = pd.Series(imputed_values[:, -1])
    data[target_col] = target_le.inverse_transform(imputed_target.astype(int))
    
    print(f"Missing values after imputation: {data[target_col].isnull().sum()}")
    return data

# Impute workclass using demographic and employment-related features
# Features: age, education, marital-status, relationship, race, gender
data = impute_column(data, 
                    'workclass', 
                    ['age', 'education', 'marital-status', 'relationship', 'race', 'gender'])

# Impute occupation using features that reflect educational background and current employment
# Features: age, education, marital-status, relationship, race, gender, workclass
data = impute_column(data, 
                    'occupation', 
                    ['age', 'education', 'marital-status', 'relationship', 'race', 'gender', 'workclass'])

# Impute native-country using demographic and socioeconomic features
# Features: age, education, race, gender, occupation, workclass, marital-status
data = impute_column(data, 
                    'native-country', 
                    ['age', 'education', 'race', 'gender', 'occupation', 'workclass', 'marital-status'])

# Impute capital-gain using work and demographic features that may affect capital income
# Features: age, educational-num, workclass, occupation, marital-status, native-country
data = impute_column(data, 
                    'capital-gain', 
                    ['age', 'educational-num', 'workclass', 'occupation', 'marital-status', 'native-country'])

# Save the results
data.to_csv('imputed_dataset.csv', index=False)
print("\nFirst few rows of final imputed data:")
print(data[['age', 'workclass', 'occupation', 'native-country', 'capital-gain']].head())



#reasons for choosing certain features for imputation:

#workclass:   Age and Education levels (both categorical and numerical) are often associated with specific types of employment sectors (e.g., younger individuals with less experience might be in entry-level jobs in the private sector).
#             Marital Status and Relationship within households can affect employment choices, as married individuals might seek more stable jobs.
#             Race and Gender can also influence work sectors due to socioeconomic factors.

#occupation:  Workclass is a key determinant of occupation since certain job roles are commonly associated with specific sectors.
#            Education levels correlate with the types of jobs individuals are qualified for.
#            Marital Status, Relationship, Race, and Gender are included due to potential impacts on occupation choice based on societal factors or family roles.

#native-country: Race and Gender can correlate with specific countries of origin, especially in countries with diverse immigrant populations.
#                Workclass and Occupation may provide insights into nationality due to differences in job distributions among immigrant populations.
#                Marital Status and Education can reflect cultural and social norms linked to nationality.


#capital-gain: Workclass and Occupation are likely related to capital gains, as certain job sectors may provide more opportunities for investments.
#             Educational Level can impact financial literacy and income, indirectly influencing capital gains.
#             Marital Status and Relationship roles can influence investment behaviors, as dual-income households may have more capital.
#             Race, Gender, and Native Country provide socioeconomic context that could impact wealth accumulation and investment trends.

# Calculating the average (mean) for each column (numerical columns only)

