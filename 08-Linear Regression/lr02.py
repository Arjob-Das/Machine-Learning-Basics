from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
plt.ion()

df = pd.read_csv('USA_Housing.csv')

print("Dataframe : {d}\n".format(d=df))
print("Head of Dataframe : {d}\n".format(d=df.head()))
print("Description of Dataframe : \n{d}".format(d=df.describe()))
print("Names of columns : {d}\n".format(d=df.columns))

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=101)
lm = LinearRegression()
lm.fit(X_train, y_train)
print("Intercept of the model : \n", lm.intercept_)
print("Coefficients of the model : \n", lm.coef_)
print("Columns of X_train : \n", X_train.columns)
print("Score of the model : \n", lm.score(X_test, y_test))
cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])
print("Coefficient Dataframe : {d}\n".format(d=cdf))
print("Head of Coefficient Dataframe : \n{d}".format(d=cdf.head()))
predictions = lm.predict(X_test)
print("Array of predictions : \n", predictions)
