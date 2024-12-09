import optuna
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Set global random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Data Preprocessing Function
def Rescaling_experiments(data, numeric_cols, columns_to_encode, scaling_method, encoding_type="onehot"):
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

    # Apply encoding to categorical columns
    if encoding_type == "onehot":
        data = pd.get_dummies(data, columns=columns_to_encode).astype(int)

    return data

# Objective Function for Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=10),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.1, step=0.005),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
        'random_state': 42  # Fixed for reproducibility
    }

    # Use all available CPU cores
    clf = RandomForestClassifier(**params, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Load Dataset
data = pd.read_csv('imputed_dataset_2.csv')

# Configuration
columns_to_encode = ['workclass', 'occupation']
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

all_results = []
for Trial in range(100, 700, 100):
    for scaling_method in scaling_methods:
        print(f"Testing Scaling Method: {scaling_methods[scaling_method]}")

        # Preprocess data
        processed_data = Rescaling_experiments(data.copy(), numeric_cols, columns_to_encode, scaling_method, encoding_type="onehot")
        print(processed_data.columns)
        x = processed_data.drop(columns=['income'])
        y = processed_data['income']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=SEED)

        # Run Optuna
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=Trial, n_jobs=1, show_progress_bar=True)

        # Best model
        best_params = study.best_params
        best_model = RandomForestClassifier(**best_params)
        best_model.fit(X_train, y_train)

        # Evaluate model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")

        print(f"Best Params: {best_params}")
        print(f"Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}")

        # Store results
        all_results.append({
            'Trial no': Trial,
            'Scaling_Method': scaling_methods[scaling_method],
            'Accuracy': accuracy,
            'Precision': precision,
            'F1': f1,
            'Recall': recall
        })

# Convert results to DataFrame
results_df = pd.DataFrame(all_results)
print("\nFinal Results Summary:")
results_df.to_csv('graphs/final_run_forest_run_2.csv', index=False)
print(results_df)