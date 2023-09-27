import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import webbrowser
import cufflinks as cf
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import __version__
import pandas as pd
import numpy as np
import chart_studio.plotly as py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
plt.ion()

train = pd.read_csv('titanic_train.csv')
print("Entire dataset : \n{d}".format(d=train))
print("Head of dataset : \n{d}".format(d=train.head()))
print("Null portions of dataset : \n{d}".format(d=train.isnull()))


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)
sex = pd.get_dummies(train['Sex'], drop_first=True)
embarck = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, sex, embarck], axis=1)
train.drop(['Sex', 'Embarked', 'Name', 'Ticket',
           'PassengerId'], axis=1, inplace=True)
print("Head of dataset : \n{d}".format(d=train.head()))

# splitting train data into test and train data set
# could have been done by importing and trimming the test csv file but this shows the process where the data isn't split into test and train
X = train.drop('Survived', axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

print("Classification Report : \n", classification_report(y_test, predictions))
print("Confusion Matrix : \n", confusion_matrix(y_test, predictions))

plt.waitforbuttonpress()
