import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()
df1 = pd.read_csv('df1', index_col=0)
print("Dataframe df1 : \n{d}".format(d=df1))
print("Head of df1 : \n{d}".format(d=df1.head()))
df2 = pd.read_csv('df2')
print("Dataframe df2 : \n{d}".format(d=df2))
print("Head of df2 : \n{d}".format(d=df2.head()))


df1['A'].hist(bins=30)
plt.pause(0.2)
df1['A'].plot(kind='hist', bins=30)
plt.pause(0.2)
df1['A'].plot.hist()
plt.pause(0.2)

df2.plot.area(alpha=0.6)  # alpha controls transparency
plt.pause(2)


plt.pause(2)
plt.waitforbuttonpress()
