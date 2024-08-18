from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../data/kc_house_data.csv')
print(df.isnull().sum())
print(df.describe().transpose())
""" plt.figure(figsize=(12, 8))
sns.displot(df['price'])
plt.pause(2)
sns.countplot(df['bedrooms'])
plt.show()
plt.pause(2)
plt.figure(figsize=(12, 8))
sns.scatterplot(x='price', y='sqft_living', data=df)
plt.pause(2)
sns.boxplot(x='bedrooms', y='price', data=df)
plt.pause(2)
plt.figure(figsize=(12, 8))
"""
""" 
sns.scatterplot(x='long', y='lat', data=df, hue='price')
plt.pause(2)
print(df.sort_values('price', ascending=False).head(20))

non_top_1_perc = df.sort_values('price', ascending=False).iloc[216:]
plt.figure(figsize=(12, 8))
sns.scatterplot(x='long', y='lat',
                data=non_top_1_perc, hue='price',
                palette='RdYlGn', edgecolor=None, alpha=0.2)
plt.pause(2)


plt.close('all')
 """
df = df.drop('id', axis=1)
print(df.head)


df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date: date.month)
df['year'] = df['date'].apply(lambda date: date.year)
print(df['year'].head)
df.groupby('month').mean()['price'].plot()

plt.pause(2)
df.groupby('year').mean()['price'].plot(
    xlim=(2014, 2015), ylim=(min(df.groupby('year').mean()['price']), max(df.groupby('year').mean()['price'])))
plt.pause(2)

df = df.drop('date', axis=1)
df = df.drop('zipcode', axis=1)


plt.close('all')


X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train.shape
X_test.shape
model = Sequential()

model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(x=X_train, y=y_train.values,
          validation_data=(X_test, y_test.values),
          batch_size=128, epochs=400)


losses = pd.DataFrame(model.history.history)
losses.plot()

print(X_test)
predictions = model.predict(X_test)
print(mean_absolute_error(y_test, predictions))
print(np.sqrt(mean_squared_error(y_test, predictions)))
print(explained_variance_score(y_test, predictions))
print(df['price'].mean())
print(df['price'].median())
# Our predictions
plt.scatter(y_test, predictions)

# Perfect predictions
plt.plot(y_test, y_test, 'r')

plt.pause(2)
errors = y_test.values.reshape(6480, 1) - predictions
sns.displot(errors)
plt.waitforbuttonpress()
