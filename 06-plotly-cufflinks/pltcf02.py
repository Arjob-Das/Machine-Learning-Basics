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

print(__version__)

data = dict(type = 'choropleth',
            locations = ['AZ','CA','NY'],
            locationmode = 'USA-states',
            colorscale= 'Portland',
            text= ['text1','text2','text3'],
            z=[1.0,2.0,3.0],
            colorbar = {'title':'Colorbar Title'})

print("Data : \n{d}".format(d=data))

layout=dict(geo={'scope':'usa'})
fig=go.Figure(data=[data],layout=layout)
plot(fig,filename='choroplot1.html')
df = pd.read_csv('2011_US_AGRI_Exports')
print("Head of Dataframe : \n{d}".format(d=df.head()))
data = dict(type = 'choropleth',
            colorscale= 'Portland',
            locations = df['code'],
            locationmode = 'USA-states',
            text= df['text'],
            z=df['total exports'],
            marker=dict(line=dict(color='rgb(255,255,255)',width=2)), #this represents the state border colour and width
            colorbar = {'title':'Millions USD'})
layout=dict(title='2011 US Agriculture Exports by State',geo=dict(scope='usa',showlakes=True,lakecolor='rgb(128,128,128)'))
fig2=go.Figure(data=[data],layout=layout)
plot(fig2,filename='choroplot2.html')
