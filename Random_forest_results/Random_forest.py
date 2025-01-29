import optuna
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score,classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer,OneHotEncoder
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate
import joblib
import pickle
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Set global random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Data Preprocessing Function
def Rescaling_experiments(data, numeric_cols, scaling_method):
    # Apply scaling to numeric columns
    if scaling_method == 0:  # No Scaling
        pass
    elif scaling_method == 1:  # MaxAbsScaler
        data[numeric_cols] = MaxAbsScaler().fit_transform(data[numeric_cols])
    elif scaling_method == 2:  # StandardScaler
        data[numeric_cols] = StandardScaler().fit_transform(data[numeric_cols])
    elif scaling_method == 3:  # MinMaxScaler
        data[numeric_cols] = MinMaxScaler().fit_transform(data[numeric_cols])
    elif scaling_method == 4:  # RobustScaler
        data[numeric_cols] = RobustScaler().fit_transform(data[numeric_cols])
    elif scaling_method == 5:  # QuantileTransformer (Uniform)
        data[numeric_cols] = QuantileTransformer(output_distribution='uniform', random_state=SEED).fit_transform(data[numeric_cols])
    elif scaling_method == 6:  # QuantileTransformer (Normal)
        data[numeric_cols] = QuantileTransformer(output_distribution='normal', random_state=SEED).fit_transform(data[numeric_cols])
    
    
   
    return data

# Objective Function for Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=100),  # Smaller range
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 10, 100),  # Smaller range
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),  # Narrowed range
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),  # Narrowed range
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True]),  # Fixed to True to simplify
        'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.01, step=0.001),  # Smaller range
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),  # Limited options
        'random_state': 42  # Fixed to ensure reproducibility
    }

    # Use cross-validation score instead of single train-test split
    clf = RandomForestClassifier(**params, n_jobs=-1)
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    
    # Return mean cross-validation score
    return scores.mean()


# Load Dataset
data = pd.read_csv('dataset.csv')

# Configuration

numeric_cols = ['age', 'educational-num', 'hours-per-week']

# Loop over different scaling methods
scaling_methods = {
    0: "No Scaling",
    1: "MaxAbsScaler",
    2: "StandardScaler",
    3: "MinMaxScaler",
    4: "RobustScaler",
    5: "QuantileTransformer (Uniform)",
    6: "QuantileTransformer (Normal)"
}



scaling_method = 0
all_results = []

"""
for Trial in range(100, 700, 100):
    for scaling_method in range(0, 7):
        print(f"Testing Scaling Method: {scaling_methods[scaling_method]}")
        
        processed_data = Rescaling_experiments(data.copy(), numeric_cols, scaling_method)
        x = processed_data.drop(columns=['income'])
        y = processed_data['income']

        # Single train-test split for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=SEED)

        # Hyperparameter optimization with cross-validation
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=Trial,  show_progress_bar=True)

        # Train final model with best parameters
        best_params = study.best_params
        best_model = RandomForestClassifier(**best_params, n_jobs=-1)
        
        # Perform k-fold cross-validation on the entire training set
        cv_scores = cross_validate(best_model, X_train, y_train, cv=5, 
                         scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])
        
        # Final evaluation on test set
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average="weighted")
        test_f1 = f1_score(y_test, y_pred, average="weighted")
        test_recall = recall_score(y_test, y_pred, average="weighted")

        # Store both cross-validation and test results
        all_results.append({
            'Trial no': Trial,
            'Scaling_Method': scaling_methods[scaling_method],
            'CV_Accuracy_Mean': cv_scores['test_accuracy'].mean(),
            'CV_Accuracy_Std': cv_scores['test_accuracy'].std(),
            'Test_Accuracy': test_accuracy,
            'Test_Precision': test_precision,
            'Test_F1': test_f1,
            'Test_Recall': test_recall,
            'Best_Params': best_params
        })

# Convert results to DataFrame
results_df = pd.DataFrame(all_results)
print("\nFinal Results Summary:")
results_df.to_csv('Dataset_3_run2.csv', index=False)
print(results_df)

"""

#####################################


print(f"Testing Scaling Method: {scaling_methods[scaling_method]}")
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
data[['age', 'educational-num', 'hours-per-week']] = np.round(scaler.inverse_transform(data[['age', 'educational-num', 'hours-per-week']]))
original_data_copy = data.copy()
processed_data = Rescaling_experiments(data.copy(), numeric_cols, scaling_method=6)
x = processed_data.drop(columns=['income'])
y = processed_data['income']
print(x.columns)


# Single train-test split for final evaluation
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=SEED)

DTC = RandomForestClassifier
#original best parameters 
best_model = DTC( 
    n_estimators=150,
    criterion='gini',
    max_depth=13,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    ccp_alpha=0.0,
    class_weight=None
   
)

best_model.fit(X_train, y_train)

# Predict the test set
prediction = best_model.predict(X_test)

# Overall performance metrics
acc_score = accuracy_score(y_test, prediction)
print(f"Overall Accuracy: {acc_score}")

# define the age ranges and features to analyse for for standard deviation 
age_ranges = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
features_to_analyse = ['educational-num', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# Iterate through age ranges
for lower, upper in age_ranges:
    print(f"\nAnalyzing age range: {lower}-{upper}")
    
    # Filter test data for the current age range
    age_filtered_data = original_data_copy.loc[X_test.index]
    age_filtered_data = age_filtered_data[(lower <= age_filtered_data['age']) & (age_filtered_data['age'] < upper)]
    print(age_filtered_data)
    if age_filtered_data.empty:
        print(f"No samples found in age range {lower}-{upper}")
        continue

    # Extract features and labels for the current age range
    X_filtered = X_test.loc[age_filtered_data.index]
    y_filtered = y_test.loc[age_filtered_data.index]

    # Predictions and performance metrics
    predictions = best_model.predict(X_filtered)
    accuracy = accuracy_score(y_filtered, predictions)
    precision = precision_score(y_filtered, predictions, average="weighted")
    recall = recall_score(y_filtered, predictions, average="weighted")
    f1 = f1_score(y_filtered, predictions, average="weighted")

    # Output metrics
    print(f"Number of samples: {len(X_filtered)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    # Calculate and output standard deviations for features
    print("\nStandard Deviations:")
    for feature in features_to_analyse:
        std = (
            y_filtered.std() if feature == 'income' 
            else age_filtered_data[feature].std()
        )
        print(f"{feature}: {std:.3f}")

for edu_level in range(1, 17):
    print(f"\nAnalyzing Educational-Num = {edu_level}")
    
    # Filter data for the current educational level
    filtered_data = original_data_copy.loc[original_data_copy.index.intersection(X_test.index)]
    filtered_data = filtered_data[filtered_data['educational-num'] == edu_level]
    if filtered_data.empty:
        print(f"No data for Educational-Num = {edu_level}")
        continue

    # Extract features and labels for the current educational level
    X_filtered = X_test.loc[filtered_data.index]
    y_filtered = y.loc[filtered_data.index]

    # Predictions and performance metrics
    predictions = best_model.predict(X_filtered)
    accuracy = accuracy_score(y_filtered, predictions)
    precision = precision_score(y_filtered, predictions, average="weighted")
    recall = recall_score(y_filtered, predictions, average="weighted")
    f1 = f1_score(y_filtered, predictions, average="weighted")

    # Output performance metrics
    print(f"Number of samples: {len(filtered_data)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    # Calculate and output standard deviations for features
    print("\nStandard Deviations:")
    for feature in features_to_analyse:
        std = y_filtered.std() if feature == 'income' else filtered_data[feature].std()
        print(f"{feature}: {std:.3f}")





# Analyze for different race groups
# Analyze performance for different race groups
race_columns = ['race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White']

for race in race_columns:
    print(f"\nAnalyzing performance for race: {race}")

    # Filter data for the current race
    filtered_data = original_data_copy.loc[X_test.index]
    filtered_data = filtered_data[filtered_data[race] == 1]

    if filtered_data.empty:
        print(f"No data for {race}")
        continue

    # Extract features and labels for the current race
    X_filtered = X_test.loc[filtered_data.index]
    y_filtered = y_test.loc[filtered_data.index]

    # Predictions and performance metrics
    predictions = best_model.predict(X_filtered)
    accuracy = accuracy_score(y_filtered, predictions)
    precision = precision_score(y_filtered, predictions, average="weighted")
    recall = recall_score(y_filtered, predictions, average="weighted")
    f1 = f1_score(y_filtered, predictions, average="weighted")

    # Output metrics
    print(f"Number of samples: {len(filtered_data)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    # Calculate and output standard deviations for features
    print("\nStandard Deviations:")
    for feature in features_to_analyse:
        std = y_filtered.std() if feature == 'income' else filtered_data[feature].std()
        print(f"{feature}: {std:.3f}")


"""
joblib.dump(best_model, 'Random_forest.joblib')

###
model = joblib.load('best_random_forest_model.joblib')

# Predict
prediction = best_model.predict([[29.0, 32.0, 10.0, 1, 0.0, 0, 60.0, 0, 0.0, 0.0, 0.0, 
                           1.0, -1.0684422957824236, 0.4608239854737111, 0.18642692922290538, 
                           -0.35369893417798676, -0.057252251562758275, -0.12196223718457576, 
                           0.003853627562193318, 0.7764769793931923, -0.05051009021640369, 
                           0.13568970638338268]])
print(prediction[0])
"""