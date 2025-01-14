import optuna
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer,OneHotEncoder
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate
import joblib

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
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500, step=100),  # Smaller range
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
        
processed_data = Rescaling_experiments(data.copy(), numeric_cols, scaling_method=6)
x = processed_data.drop(columns=['income'])
y = processed_data['income']


print(x.head())
print(y.head())
# Single train-test split for final evaluation
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=SEED)



DTC=RandomForestClassifier
best_model = DTC(
    n_estimators=1400,         # Number of trees in the forest
    criterion='entropy',       # Split quality: 'entropy' for information gain
    max_depth=96,              # Maximum depth of each tree
    min_samples_split=50,      # Minimum number of samples required to split an internal node
    min_samples_leaf=1,        # Minimum number of samples required to be at a leaf node
    max_features='sqrt',       # Number of features to consider for splitting at each node ('sqrt' = square root of features)
    bootstrap=True,            # Whether bootstrap samples are used when building trees
    ccp_alpha=0.0,             # Complexity parameter for pruning (0.0 = no pruning)
    class_weight=None,         # Weights associated with classes (None = all classes are weighted equally)
    random_state=42            # Seed for reproducibility
)


best_model.fit(X_train,y_train)
###########################################################
#predict the test set
prediction=best_model.predict(X_test)

acc_score = accuracy_score(y_test, prediction)
print(f"Accuracy: {acc_score}")
joblib.dump(best_model, 'best_random_forest_model.joblib')

###
model = joblib.load('best_random_forest_model.joblib')

# Predict
prediction = best_model.predict([[29.0, 32.0, 10.0, 1, 0.0, 0, 60.0, 0, 0.0, 0.0, 0.0, 
                           1.0, -1.0684422957824236, 0.4608239854737111, 0.18642692922290538, 
                           -0.35369893417798676, -0.057252251562758275, -0.12196223718457576, 
                           0.003853627562193318, 0.7764769793931923, -0.05051009021640369, 
                           0.13568970638338268]])
print(prediction[0])
