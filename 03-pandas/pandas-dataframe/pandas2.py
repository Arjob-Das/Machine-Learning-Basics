import numpy as np
import pandas as pd

from numpy.random import randn
np.random.seed(101)

df = pd.DataFrame(randn(5, 4), ['A', 'B', 'C', 'D', 'E'], ['W', 'X', 'Y', 'Z'])

print("Data Frame : \n", df)

print("W column in Data Frame : \n", df['W'])
print(df.W)

print("Multiple Columns : \n", df[['W', 'Z']])

df['new'] = df['W']+df['Y']
print("Modified DataFrame : \n", df)

print("Displaying temorarily dropped new column : \n", df.drop(
    'new', axis=1))  # axis 1 is for columns and axis 0 for rows
print("Showing that df reamins unchanged by previous action : \n", df)

df = df.drop('new', axis=1)
print("After dropping df permanently : \n", df)

# another way to permanently drop : using inplace argument : df.drop('new',axis=1,inplace=True)
df.drop('E', axis=0, inplace=True)
print("Displaying permanently dropped E row : \n", df)

# accessing rows in dataframes
print("Row A : ")
print(df.loc['A'])
print("Accessing Row C using numerical index position : ")
print(df.iloc[2])

# selecting subsets of rows and columns:
print("Data from Row B Column Y : ")
print(df.loc['B', 'Y'])

print("Data from Rows A,B Columns W,Y : ")
print(df.loc[['A', 'B'], ['W', 'Y']])

df = pd.DataFrame(randn(5, 4), ['A', 'B', 'C', 'D', 'E'], ['W', 'X', 'Y', 'Z'])

print("DataFrame : ")
print(df)
