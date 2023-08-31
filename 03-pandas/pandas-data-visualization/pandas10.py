import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
df1 = pd.read_csv('df1', index_col=0)
print("Dataframe df1 : \n{d}".format(d=df1))
print("Head of df1 : \n{d}".format(d=df1.head()))
df2 = pd.read_csv('df2')
print("Dataframe df2 : \n{d}".format(d=df2))
print("Head of df2 : \n{d}".format(d=df2.head()))
