import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Load the dataset
file_path = 'adult.csv'
data = pd.read_csv(file_path)

# Replace both '?' and 99999 with np.nan
data.replace(["?", 99999], np.nan, inplace=True)

# One-hot encode categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Impute missing values using KNNImputer
imputer = KNNImputer(n_neighbors=5)
data_imputed = imputer.fit_transform(data)

# Convert the imputed data back to a DataFrame
data_imputed = pd.DataFrame(data_imputed, columns=data.columns)

# Save the results
data_imputed.to_csv('imputed_dataset.csv', index=False)
print("\nFirst few rows of final imputed data:")
print(data_imputed.head())



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

