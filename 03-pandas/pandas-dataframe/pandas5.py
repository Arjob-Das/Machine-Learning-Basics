import numpy as np
import pandas as pd

from numpy.random import randn

d={'A':[1,2,np.nan],'B':[5,np.nan,np.nan],'C':[1,2,3]}

df=pd.DataFrame(d)

print("Initial Dataframe : \n",df)

print("DataFrame after dropping all rows with null values : \n",df.dropna()) #axis is 0 by default signifying rows

print("DataFrame after dropping entire columns which have null values : \n",df.dropna(axis=1))

print("Dataframe after dropping rows that don't have atleast 2 non-null values : \n",df.dropna(thresh=2))

print("Filling Values in null values positions : \n",df.fillna(value='Fill Value'))
print("Dataframe after filling without using in-place argument : \n",df)
""" print("Filling Values in null values positions : \n",df.fillna(value='Fill Value',inplace=True))
print("Dataframe after filling and using in-place argument : \n",df)
 """
print("Filling Values with mean of current DataFrame in null values positions : \n",df.fillna(value=df.mean()))
print("Filling Values of Column A with mean of current A Column in null values positions : \n",df['A'].fillna(value=df['A'].mean()))
print("Dataframe after filling without using in-place argument : \n",df)


