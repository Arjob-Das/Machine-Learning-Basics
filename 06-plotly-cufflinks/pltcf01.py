import plotly.express as px
import webbrowser
import cufflinks as cf
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import __version__
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
cf.go_offline()
print(__version__)  # requires version >= 1.9.0
df = pd.DataFrame(np.random.randn(100, 4), columns='A B C D'.split())
print("Head of Dataframe : \n{d}".format(d=df.head()))
df2 = pd.DataFrame({'Category': ['A', 'B', 'C'], 'Values': [32, 43, 50]})

print("Head of Dataframe : \n{d}".format(d=df2.head()))

""" df3 = pd.DataFrame(np.random.randn(90000000, 4), columns='A B C D'.split())
print("Head of Dataframe :{d} \n".format(d=df3)) """

""" 
to achive the following : 
df.iplot(kind='scatter',x='A',y='B',mode='markers',size=10)
"""
fig1 = go.Figure(data=go.Scatter(
    x=df['A'], y=df['B'], mode='markers', marker=dict(size=10, color='blue', showscale=True)))
# fig1.show()
""" fig1 = go.Figure(data=go.Scatter(
    x=df3['A'], y=df3['B'], mode='markers', marker=dict(size=10, color='blue', showscale=True)))
fig1.show() """
""" df.iplot(kind='scatter', x='A', y='B', mode='markers', size=10)
#works only for notebooks
# """
fig1 = go.Figure(data=go.Bar(
    x=df2['Category'], y=df2['Values']))
# fig1.show()
""" 
to achive the following 
df.count().iplot(kind='bar')
df.sum().iplot(kind='bar')
"""
fig1 = go.Figure(data=go.Bar(
    y=df.sum(), x=df.columns))
# fig1.show()
fig1 = go.Figure(data=go.Bar(
    y=df.count(), x=df.columns))
# fig1.show()
fig1 = go.Figure(data=go.Box(
    y=df))
# fig1.show()

# another method

fig = go.Figure()

# Add separate box plots for each column
for column in df.columns:
    fig.add_trace(go.Box(y=df[column], name=column))

# Customize the layout
fig.update_layout(
    title='Box Plots for DataFrame Columns',
    # xaxis=dict(title='Category'),
    # yaxis=dict(title='Value')
)
# fig.show()
X = list(np.random.randint(1, 100, 10))
Y = list(np.random.randint(1, 100, 10))
Z = [list(np.random.randint(1, 100, 10)) for _ in range(10)]
print(Z)
surface_plot = go.Surface(x=X, y=Y, z=Z)
layout = go.Layout(
    title='3D Surface Plot Example',
    scene=dict(
        xaxis_title='X-Axis',
        yaxis_title='Y-Axis',
        zaxis_title='Z-Axis'
    )
)

# Create a figure
fig = go.Figure(data=[surface_plot], layout=layout)
plot(fig, filename='3d_surface_plot.html')

df3 = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

surface_plot = go.Surface(
    x=df3['X'], y=df3['Y'], z=df3['Z'], colorscale='Viridis')
layout = go.Layout(
    title='3D Surface Plot Example',
    scene=dict(
        xaxis_title='X-Axis',
        yaxis_title='Y-Axis',
        zaxis_title='Z-Axis'
    )
)

# Create a figure
fig = go.Figure(data=[surface_plot], layout=layout)
fig.show()
