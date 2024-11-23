##impor the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier


def prepare_dataset(data):
    X = data.drop(columns=['income'])
    y = data['income']

    # Encode gender first
    gender_encoder = LabelEncoder()
    X['gender'] = gender_encoder.fit_transform(X['gender'])

    #perform one hot encoding on the categorical features
    onehot_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    ct = ColumnTransformer([('onehot', OneHotEncoder(drop='first', sparse_output=False), onehot_cols)], remainder='passthrough')

    X = np.array(ct.fit_transform(X))

    # Encode target variable
    y_encoder = LabelEncoder()
    encoded_y = y_encoder.fit_transform(y)

    feature_names = ct.get_feature_names_out()

    #split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, encoded_y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test,feature_names

    #function used to test the best parameters for the decision tree
def Test_best_paramters(X_train, y_train):
    print("Testing the best parameters")
    param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2', None]
    }

    # Create scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted')
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring=scoring,
        refit='accuracy',  # Use accuracy to select the best model
        n_jobs=-1,  # Use all available cores
        verbose=2
)

    # Fit the grid search
    grid_search.fit(X_train, y_train)
    # Print the best parameters and score
    print("\nBest parameters:", grid_search.best_params_)
    print("Best cross-validation accuracy: {:.3f}".format(grid_search.best_score_))

    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Add this line to return the best model
    return best_model  # <-- Add this line
def train_decision_tree(X_train, y_train):
    # Evaluate on test set
    best_model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=10,
        max_features=None,
        min_samples_leaf=2,
        min_samples_split=10,
        random_state=42
    )
    best_model.fit(X_train, y_train)
    return best_model
def predict_model_decision_tree(model, X_test, y_test, y_encoder):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nTest set accuracy with best model: {:.3f}".format(accuracy))

    # Create and plot confusion matrix for the best model
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                display_labels=y_encoder.classes_)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix (Best Model)')
    plt.show()

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=y_encoder.classes_))


def train_random_forest_tree(X_train, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
    )
    rf_model.fit(X_train, y_train)
    return rf_model
def predict_random_forest_tree(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nTest set accuracy with best model: {:.3f}".format(accuracy))
def fine_tune_decision_tree(X_train, y_train):
    print("Starting Decision Tree fine-tuning...")
    
    param_grid = {
        'max_depth': [3, 5, 7, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', None],
        'splitter': ['best', 'random']
    }

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted')
    }

    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        refit='accuracy',
        n_jobs=-1,
        verbose=1
    )

    print("Fitting GridSearchCV... This might take a while...")
    grid_search.fit(X_train, y_train)

    print("\nBest parameters found:", grid_search.best_params_)
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.3f}")

    return grid_search.best_estimator_
def analyze_feature_importance(model, feature_names=None):
    print(feature_names)
    feature_importance = pd.DataFrame({
        'feature': feature_names if feature_names is not None else range(len(model.feature_importances_)),
        'importance': model.feature_importances_
    })
    feature_importance['importance'] = feature_importance['importance'].round(2)  # Round to two decimal places
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Importance (rounded to 2 decimal places)')  # Update x-axis label
    plt.ylabel('Features')  # Update y-axis label
    plt.show()

def fine_tune_random_forest(X_train, y_train):
    print("Starting Random Forest fine-tuning...")
    
    # Define parameter grid for Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],  # number of trees
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted')
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        refit='accuracy',
        n_jobs=-1,
        verbose=1
    )

    print("Fitting GridSearchCV... This might take a while...")
    grid_search.fit(X_train, y_train)

    print("\nBest parameters found:", grid_search.best_params_)
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.3f}")

    return grid_search.best_estimator_

#load the dataset
file_path = 'complete_dataset.csv'
data = pd.read_csv(file_path)

X_train,X_test,y_train,y_test,feature_names = prepare_dataset(data)
#split the features and target




#best_model = Test_best_paramters(X_train, y_train)
#Trained_decision_tree = train_decision_tree(X_train, y_train)
#predict_model_decision_tree(Trained_decision_tree, X_test, y_test, y_encoder)

#best_estimator_random_forest = fine_tune_random_forest(X_train, y_train)
Trained_random_forest_tree = train_random_forest_tree(X_train, y_train)
#analyze_feature_importance(model=Trained_random_forest_tree, feature_names=feature_names)
predict_random_forest_tree(Trained_random_forest_tree, X_test, y_test)



