import plotly.express as px
import webbrowser
import cufflinks as cf
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import __version__
import pandas as pd
import numpy as np
import chart_studio.plotly as py
import matplotlib.pyplot as plt
plt.ion()
cf.go_offline()

df = pd.read_csv('2014_World_GDP')
print("Head of Dataframe : \n{d}".format(d=df.head()))
data = dict(type = 'choropleth',
            #colorscale= 'Portland',
            locations = df['CODE'],
            text= df['COUNTRY'],
            z=df['GDP (BILLIONS)'],
            #marker=dict(line=dict(color='rgb(255,255,255)',width=2)), #this represents the state border colour and width
            colorbar = {'title':'GDP in Billions USD'})
layout=dict(title='2014 Global GDP',geo=dict(showframe=False,projection={'type':'mercator'}))
fig3=go.Figure(data=[data],layout=layout)
plot(fig3,filename='choroplot3.html')
