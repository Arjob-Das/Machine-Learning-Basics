import numpy as np
import pandas as pd

from numpy.random import randn

d = {'A': [1, 2, np.nan], 'B': [5, np.nan, np.nan], 'C': [1, 2, 3]}

df = pd.DataFrame(d)

print("Initial Dataframe : \n", df)

print("DataFrame after dropping all rows with null values : \n",
      df.dropna())  # axis is 0 by default signifying rows

print("DataFrame after dropping entire columns which have null values : \n",
      df.dropna(axis=1))

print("Dataframe after dropping rows that don't have atleast 2 non-null values : \n",
      df.dropna(thresh=2))

print("Filling Values in null values positions : \n",
      df.fillna(value='Fill Value'))
print("Dataframe after filling without using in-place argument : \n", df)
""" print("Filling Values in null values positions : \n",df.fillna(value='Fill Value',inplace=True))
print("Dataframe after filling and using in-place argument : \n",df)
 """
print("Filling Values with mean of current DataFrame in null values positions : \n",
      df.fillna(value=df.mean()))
print("Filling Values of Column A with mean of current A Column in null values positions : \n",
      df['A'].fillna(value=df['A'].mean()))
print("Dataframe after filling without using in-place argument : \n", df)

""" 
Initial Dataframe : 
      A    B  C
0  1.0  5.0  1
1  2.0  NaN  2
2  NaN  NaN  3
DataFrame after dropping all rows with null values : 
      A    B  C
0  1.0  5.0  1
DataFrame after dropping entire columns which have null values :
    C
0  1
1  2
2  3
Dataframe after dropping rows that don't have atleast 2 non-null values :
      A    B  C
0  1.0  5.0  1
1  2.0  NaN  2
Filling Values in null values positions :
             A           B  C
0         1.0         5.0  1
1         2.0  Fill Value  2
2  Fill Value  Fill Value  3
Dataframe after filling without using in-place argument :
      A    B  C
0  1.0  5.0  1
1  2.0  NaN  2
2  NaN  NaN  3
Filling Values with mean of current DataFrame in null values positions :
      A    B  C
0  1.0  5.0  1
1  2.0  5.0  2
2  1.5  5.0  3
Filling Values of Column A with mean of current A Column in null values positions : 
 0    1.0
1    2.0
2    1.5
Name: A, dtype: float64
Dataframe after filling without using in-place argument :
      A    B  C
0  1.0  5.0  1
1  2.0  NaN  2
2  NaN  NaN  3
"""
