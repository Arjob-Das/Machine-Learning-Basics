from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
plt.ion()

df = pd.read_csv("Classified Data", index_col=0)
print("Dataframe : \n{d}".format(d=df))
print("Head of Dataframe : \n{d}".format(d=df.head()))
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))

scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
print("Array object after scaling : \n{d}".format(d=scaled_features))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
print("Head of Dataframe after scaling : \n{d}".format(d=df_feat.head()))

X = df_feat  # or scaled_features
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

""" for i in range(1, 50):
    print("For {n} neighbors".format(n=i))
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))
 """

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

error_rate = []

for i in range(1, 40):
    print("For {n} neighbors".format(n=i))
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    print("Confusion matrix : \n{d}".format(d=confusion_matrix(y_test, pred)))
    print("Classification report : \n{d}".format(
        d=classification_report(y_test, pred)))
plt.figure(figsize=(10, 6))
print("Error Rate : \n{d}".format(d=error_rate))
plt.plot(range(1, 40), error_rate, color='blue',
         linestyle='dashed', marker='o',)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.pause(2)

error_rate = []

for i in range(30, 300):
    # print("For {n} neighbors".format(n=i))
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    """ print("Confusion matrix : \n{d}".format(d=confusion_matrix(y_test, pred)))
    print("Classification report : \n{d}".format(
        d=classification_report(y_test, pred))) """
plt.figure(figsize=(16, 9))
print("Error Rate : \n{d}".format(d=error_rate))
plt.plot(range(30, 300), error_rate, color='blue',
         linestyle='dashed', marker='o',)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.pause(20)
print("Lowest Error Rate : {n}, {m}".format(
    n=error_rate.index(min(error_rate)), m=min(error_rate)))

# plt.waitforbuttonpress()
