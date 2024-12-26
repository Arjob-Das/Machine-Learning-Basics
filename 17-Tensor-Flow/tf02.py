
from tensorflow.keras.layers import Dense, Dropout
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

df = pd.read_csv('DATA/cancer_classification.csv')
print("Head of data : \n{d}".format(d=df.head()))

print("Description of data : \n{d}".format(d=df.describe().transpose()))

sns.countplot(x='benign_0__mal_1', data=df, palette="Set1")

plt.pause(2)
plt.close('all')
df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')

plt.pause(2)

plt.figure(figsize=(12, 12))
sns.heatmap(df.corr())

plt.pause(2)
plt.waitforbuttonpress(1)

X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=101)
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
model = Sequential()

model.add(Dense(30, activation='relu'))

model.add(Dense(15, activation='relu'))

# Binary Classification, so acitvation for last layer is sigmoid
model.add(Dense(1, activation='relu'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test))
