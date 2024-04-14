from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv(
    'F:/MyDrive/Self Study/ML Basics/ML-Basics/15-Recommender-Systems/ratings.csv')
print("Head of data : \n{d}".format(d=df.head()))
n_users = df.user_id.nunique()
n_items = df.item_id.nunique()

print('Num. of Users: ' + str(n_users))
print('Num of Movies: '+str(n_items))

unique_user_ids = df['user_id'].unique()

mapping_dict = dict(zip(unique_user_ids, range(1, len(unique_user_ids) + 1)))

df['user_id'] = df['user_id'].map(mapping_dict)

df.reset_index(drop=True, inplace=True)

df = df[df['user_id'] < (n_users/8)]
print("Head of dataset : \n{d}".format(d=df.head()))
print("Shape of dataset : \n{d}".format(d=df.shape))


unique_item_ids = df['item_id'].unique()

mapping_dict = dict(zip(unique_item_ids, range(1, len(unique_item_ids) + 1)))

df['item_id'] = df['item_id'].map(mapping_dict)

df.reset_index(drop=True, inplace=True)

df = df[df['item_id'] < (n_items/8)]
print("Head of dataset : \n{d}".format(d=df.head()))
print("Shape of dataset : \n{d}".format(d=df.shape))
X = df[['user_id', 'item_id']].values
y = df['rating'].values

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
    Dense(8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(4, activation='relu'),
    Dense(4, activation='relu'),
    # Dense(8, activation='relu'),
    # Dense(4, activation='relu'),
    # Dense(2, activation='relu'),
    Dense(1)
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=20, min_delta=0.001)

# Train the model
history = model.fit(X_train, y_train, epochs=10,
                    steps_per_epoch=100, use_multiprocessing=True, callbacks=[early_stopping])

# loss_df = pd.DataFrame(model.history.history)
""" loss_df.plot()
plt.waitforbuttonpress()
 """
# model evaluation methods

# method 1
print("Loss of model on test set \n", model.evaluate(X_test, y_test, verbose=0))
# print("Loss of model on train set \n",model.evaluate(X_train, y_train, verbose=0))
test_predictions = model.predict(X_test)
# print(test_predictions)

test_predictions = pd.Series(
    test_predictions.reshape(test_predictions.shape[0],))
# print(test_predictions)

pred_df = pd.DataFrame(y_test, columns=['Test True Y'])
pred_df['Model Predictions'] = test_predictions
print("Dataframe pred_df : \n{d}".format(d=pred_df))
