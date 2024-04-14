from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('../DATA/diamonds.csv')
print("Head of data : \n{d}".format(d=df.head()))

label_encoder = LabelEncoder()
df['cut'] = label_encoder.fit_transform(df['cut'])
df['color'] = label_encoder.fit_transform(df['color'])
df['clarity'] = label_encoder.fit_transform(df['clarity'])

print("Head of data : \n{d}".format(d=df.head()))
df = df[(df[['x', 'y', 'z']] != 0).all(axis=1)]

df['volume'] = df['x'] * df['y'] * df['z']
print("Head of data : \n{d}".format(d=df.head()))

X = df[['clarity', 'color', 'cut', 'carat',
        'depth', 'table', 'volume']].values
y = df['price'].values

print("X and y : \n{d1} \n{d2}".format(d1=X, d2=y))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

print("Shape of X_train : \n{d}".format(d=X_train.shape))
print("Shape of X_test : \n{d}".format(d=X_test.shape))

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
model = Sequential([
    # Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(2, activation='relu'),
    Dense(1)
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=15, min_delta=0.001)

# Train the model
history = model.fit(X_train, y_train, epochs=1200,
                    steps_per_epoch=100, use_multiprocessing=True, callbacks=[early_stopping])

loss_df = pd.DataFrame(model.history.history)
print("Loss of model on test set \n", model.evaluate(X_test, y_test, verbose=0))
print("Loss of model on train set \n",
      model.evaluate(X_train, y_train, verbose=0))
test_predictions = model.predict(X_test)
# print(test_predictions)

test_predictions = pd.Series(
    test_predictions.reshape(test_predictions.shape[0],))
# print(test_predictions)

pred_df = pd.DataFrame(y_test, columns=['Test True Y'])
pred_df['Model Predictions'] = test_predictions
print("Dataframe pred_df : \n{d}".format(d=pred_df))
