from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('../DATA/fake_reg.csv')
print("Head of data : \n{d}".format(d=df.head()))

""" sns.pairplot(df)
plt.waitforbuttonpress() """
X = df[['feature1', 'feature2']].values
y = df['price'].values

print("X and y : \n{d1} \n{d2}".format(d1=X, d2=y))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

print("Shape of X_train : \n{d}".format(d=X_train.shape))
print("Shape of X_test : \n{d}".format(d=X_test.shape))

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

""" #method 1
model=Sequential([Dense(4,activation='relu'),Dense(2,activation='relu'),Dense(1)])
 """
# method 2 preferred so that layers can be commented out or edited easily
model = Sequential()
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse')

early_stopping = EarlyStopping(monitor='loss', patience=50, min_delta=0.001)
# the epoches stop running if there is no improvement in loss for 5 (patience) consecutive epochs
model.fit(x=X_train, y=y_train, epochs=2500, callbacks=[
          early_stopping], use_multiprocessing=True, steps_per_epoch=50)

loss_df = pd.DataFrame(model.history.history)
""" loss_df.plot()
plt.waitforbuttonpress()
 """
# model evaluation methods

# method 1
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

sns.scatterplot(x='Test True Y', y='Model Predictions', data=pred_df)
plt.pause(2)
# plt.waitforbuttonpress()
plt.close('all')
print("Mean Absolute Error : \n", mean_absolute_error(
    pred_df['Test True Y'], pred_df['Model Predictions']))
print("Mean Squared Error : \n", mean_squared_error(
    pred_df['Test True Y'], pred_df['Model Predictions']))
print("R2 Score : \n", r2_score(
    pred_df['Test True Y'], pred_df['Model Predictions']))

new_gem = [[998, 1000]]
new_gem = scaler.transform(new_gem)
print("New gem prediction : \n", model.predict(new_gem))

model.save('my_gem_model.h5')
later_model = load_model('my_gem_model.h5')
print("New gem prediction with later model: \n", later_model.predict(new_gem))
