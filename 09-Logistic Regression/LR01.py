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
""" sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.pause(2)
sns.set_style('whitegrid')

plt.figure()
sns.countplot(x='Survived', data=train)
plt.pause(2)
plt.figure()
sns.countplot(x='Survived', data=train, hue='Sex')
plt.pause(2)
plt.close('all')

sns.countplot(x='Survived', data=train, hue='Pclass')
plt.pause(2)
sns.displot(train['Age'].dropna(), bins=30)
plt.pause(2)
plt.figure()
train['Age'].plot.hist(bins=35)
plt.pause(2)
plt.close('all')

print("Info of dataset : \n{d}".format(d=train.info()))
sns.countplot(x='SibSp', data=train)
plt.pause(2)
plt.figure()
train['Fare'].hist(bins=40, figsize=(10, 4))
plt.pause(2)
plt.close('all') """

cf.go_offline()
trace = go.Histogram(x=train['Fare'])
data = [trace]

layout = go.Layout(title='Histogram Plot')
fig = go.Figure(data=data, layout=layout)
# iplot(fig)
plot(fig, filename='histogram_iplot_of_fare.html')

plt.waitforbuttonpress()
