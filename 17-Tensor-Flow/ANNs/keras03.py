import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from keras.models import load_model


model = load_model('house_price_prediction_model.h5')

df = pd.read_csv('../data/kc_house_data.csv')
print(df.isnull().sum())
print(df.describe().transpose())
df = df.drop('id', axis=1)
print(df.head)


df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date: date.month)
df['year'] = df['date'].apply(lambda date: date.year)

df = df.drop('date', axis=1)
df = df.drop('zipcode', axis=1)

X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape)
print(X_test.shape)
print(X_test)
predictions = model.predict(X_test)
print(mean_absolute_error(y_test, predictions))
print(np.sqrt(mean_squared_error(y_test, predictions)))
print(explained_variance_score(y_test, predictions))
print(df['price'].mean())
print(df['price'].median())


with open('house_price_prediction_model_history.pkl', 'rb') as file:
    history = pickle.load(file)
losses = pd.DataFrame(history)
print(losses)

losses.plot()

plt.pause(2)

plt.close('all')

predictions = model.predict(X_test)

errors = y_test.values.reshape(6480, 1) - predictions

print(explained_variance_score(y_test, predictions))

plt.figure(figsize=(12, 6))
plt.scatter(y_test, predictions)
plt.plot(y_test, y_test, 'r')
# the outlying expensive houses cause error mostly
plt.pause(2)

print(df.drop('price', axis=1).iloc[0])
single_house = df.drop('price', axis=1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1, 19))
print(model.predict(single_house))


# plt.waitforbuttonpress()
