from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.ion()

df = pd.read_csv('kyphosis.csv')
print("Head of the dataset : \n{d}".format(d=df.head()))
print("Information of the dataset : \n{d}".format(d=df.info()))
sns.pairplot(df, hue='Kyphosis')
plt.waitforbuttonpress()

X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
print("Confusion Matrix : \n{d}".format(
    d=confusion_matrix(y_test, predictions)))
print("Classification Report : \n{d}".format(
    d=classification_report(y_test, predictions)))

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
rfc_predictions = rfc.predict(X_test)

print("Confusion Matrix : \n{d}".format(
    d=confusion_matrix(y_test, rfc_predictions)))
print("Classification Report : \n{d}".format(
    d=classification_report(y_test, rfc_predictions)))

