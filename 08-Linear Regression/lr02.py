from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
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

print("Dataframe : \n{d}".format(d=df))
print("Head of Dataframe : \n{d}".format(d=df.head()))
print("Description of Dataframe : \n{d}".format(d=df.describe()))
print("Names of columns : \n{d}".format(d=df.columns))

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
print("Coefficient Dataframe : \n{d}".format(d=cdf))
print("Head of Coefficient Dataframe : \n{d}".format(d=cdf.head()))
predictions = lm.predict(X_test)
print("Array of predictions : \n", predictions)
print("Correct predictions : \n", y_test)
plt.scatter(y_test, predictions)
plt.pause(2)
sns.displot((y_test-predictions), kde=True)
plt.pause(2)
print("Mean Absolute Error : \n", metrics.mean_absolute_error(y_test, predictions))
print("Mean Squared Error : \n", metrics.mean_squared_error(y_test, predictions))
print("Root Mean Squared Error : \n",
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))
plt.close('all')

calf = fetch_california_housing()
print(calf.keys())
print(calf['DESCR'])
print(calf['feature_names'])
print(calf['data'])


# Load the California housing data
data = fetch_california_housing(as_frame=True)

# Create a DataFrame with the feature data
df = data.data

# Add the target variable to the DataFrame
df['target'] = data.target

# Split the data into features (X) and target variable (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Perform linear regression, evaluate the model, and make predictions (steps 4-6 from the previous answer)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

plt.scatter(y_test, y_pred)
plt.pause(2)
sns.displot((y_test-y_pred), kde=True, bins=100)
plt.pause(2)
plt.waitforbuttonpress()
