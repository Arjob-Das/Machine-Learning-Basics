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

plt.figure(figsize=(10, 7))
sns.boxplot(x='Pclass', y='Age', data=train)
plt.pause(2)


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


print("Age column before updating : \n{d}".format(d=train['Age']))
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
print("Age column after updating : \n{d}".format(d=train['Age']))

plt.figure()
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.pause(2)

train.drop('Cabin', axis=1, inplace=True)
print("Head of dataset : \n{d}".format(d=train.head()))
train.dropna(inplace=True)
print("Head of dataset : \n{d}".format(d=train.head()))
plt.figure()
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.pause(2)
plt.close('all')

# for machine learning algorithm to understand string labels must be converted to dummy variables with numerical values

# pd.get_dummies(train['Sex'])
# this creates a table with 2 columns where 1 is true and 0 is false
# however if somone is not male it can be considered that the person is female, so only one column is needed
sex = pd.get_dummies(train['Sex'], drop_first=True)
embarck = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, sex, embarck], axis=1)
print("Head of dataset : \n{d}".format(d=train.head()))

# the sex and embarked columns are no longer needed, the name and ticket are also not needed
# passengerID is also dropped as it is simply the index and has no relation to the algorithm

train.drop(['Sex', 'Embarked', 'Name', 'Ticket',
           'PassengerId'], axis=1, inplace=True)
print("Head of dataset : \n{d}".format(d=train.head()))


plt.waitforbuttonpress()
