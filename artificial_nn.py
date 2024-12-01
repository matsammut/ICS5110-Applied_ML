import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import decision_tree as dt

import optuna

scaler = StandardScaler()
le = LabelEncoder()
data = pd.read_csv('imputed_dataset.csv')

X_train, X_test, y_train, y_test,feature_names,encoded_y = dt.prepare_dataset(data)


def objective(trial):
    neurons = trial.suggest_int("neurons", 8, 64, log=True)
    layers = trial.suggest_int("layers", 1, 8, log=True)
    activation = trial.suggest_categorical("activation_function", ['sigmoid', 'relu',  'softmax'])
    epochs = trial.suggest_int("epochs", 10, 150, log=True)
    dropout = trial.suggest_uniform("dropout_rate", 0, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)

    ANNmodel = Sequential()
    ANNmodel.add(Dense(neurons, input_dim=20, activation=activation))
    while layers > 0:
        ANNmodel.add(Dense(neurons, activation=activation))
        ANNmodel.add(Dropout(dropout))
        layers -= 1
    ANNmodel.add(Dense(1, activation='sigmoid'))

    # binary_crossentropy is used instead of categorical_crossentropy because there are only two catagories male/female if we had more we would have had to use categorical
    ANNmodel.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
    ANNmodel.fit(x_train2, y_train2, epochs=epochs, batch_size=32, verbose=0)
    score = ANNmodel.evaluate(x_test2, y_test2)
    return score[1]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=250)

trial = study.best_trial
best_params = trial.params

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

fig = optuna.visualization.plot_optimization_history(study)
fig.show()
fig2 = optuna.visualization.plot_slice(study)
fig2.show()

ANNmodel = Sequential()
ANNmodel.add(Dense(best_params["neurons"], input_dim=20, activation=best_params["activation_function"]))
layers = best_params["layers"]
while layers > 0:
    ANNmodel.add(Dense(best_params["neurons"], activation=best_params["activation_function"]))
    ANNmodel.add(Dropout(best_params["dropout_rate"]))
    layers -= 1
ANNmodel.add(Dense(1, activation='sigmoid'))

# binary_crossentropy is used instead of categorical_crossentropy because there are only two catagories male/female if we had more we would have had to use categorical
ANNmodel.compile(loss='binary_crossentropy', optimizer=Adam(best_params["learning_rate"]), metrics=['accuracy'])

ANNmodel.fit(x_train2, y_train2, epochs=best_params["epochs"], batch_size=32, verbose=0)
# Making predictions
y_pred2 = ANNmodel.predict(x_test2)
y_pred2num = (y_pred2 > 0.5).astype(int)

# Printing Results
print('score of Neural Network model is: ', ANNmodel.evaluate(x_test2, y_test2)[1])
print("\t\t\tNeural Network Class report:\n", classification_report(y_pred2num, y_test2))
print("Neural Network Accuracy score: ", accuracy_score(y_pred2num, y_test2) * 100, "%")

y_pred2 = np.where(y_pred2 > 0.5, 'male', 'female')
y_pred2 = np.squeeze(y_pred2)
print(y_pred2[:10])
pd.crosstab(y_pred2, y_test2, rownames=['matrix'], colnames=['confusion'], margins=True)
