from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
plt.ion()

cancer = load_breast_cancer()
print("cancer.keys(): \n{d}".format(d=cancer.keys()))
print("Description of the dataset : \n{d}".format(d=cancer['DESCR']))
df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
print("Head of df_feat dataset : \n{d}".format(d=df_feat.head()))
print("Target names of cancer dataset : \n{d}".format(
    d=cancer['target_names']))
df_target = pd.DataFrame(cancer['target'], columns=['Cancer'])
print("Head of df_target dataset : \n{d}".format(d=df_target.head()))

X = df_feat
Y = df_target
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=101)
model = SVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("1st Classification Report : \n{d}".format(
    d=classification_report(y_test, predictions)))
print("1st Confusion Matrix : \n{d}".format(
    d=confusion_matrix(y_test, predictions)))

X = df_feat
Y = np.ravel(df_target)  # to remove column 1d array warning
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=101)
model = SVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("2nd Classification Report : \n{d}".format(
    d=classification_report(y_test, predictions)))
print("2nd Confusion Matrix : \n{d}".format(
    d=confusion_matrix(y_test, predictions)))

# from sklearn.grid_search import GridSearchCV -> depriciated

param_grid = {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)
print("Best Parameters : \n{d}".format(d=grid.best_params_))
print("Best Estimator : \n{d}".format(d=grid.best_estimator_))
grid_predictions = grid.predict(X_test)
print("3rd Classification Report : \n{d}".format(d=classification_report(
    y_test, grid_predictions)))
print("3rd Confusion Matrix : \n{d}".format(d=confusion_matrix(
    y_test, grid_predictions)))
