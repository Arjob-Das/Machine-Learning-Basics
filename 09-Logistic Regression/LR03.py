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



plt.waitforbuttonpress()
