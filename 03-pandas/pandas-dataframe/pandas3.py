import numpy as np
import pandas as pd

from numpy.random import randn
np.random.seed(101)

df = pd.DataFrame(randn(5, 4), ['A', 'B', 'C', 'D', 'E'], ['W', 'X', 'Y', 'Z'])

print("DataFrame : ")
print(df)

# conditional selection

print("Dataframe boolean for values greater than 0 : ")
print(df > 0)

booldf = df > 0

print("The Boolean Dataframe : ")
print(booldf)

# combining boolean dataframe to get conditionally selected values

print("Boolean dataframe to get conditionally selected values : ")
print("Method 1 :  using the booldf : ")
print(df[booldf])
print("Method 2 : Using the conditon directly : ")
print(df[df > 0])

# using above on subsets
print("Rows where values in Column W are greater than 0 : ")
print(df[df['W'] > 0])

print("Rows where values in Column Z are less than 0 : ")
print(df[df['Z'] < 0])

# selecting specific results from conditionally selected dataframe:

resdf = df[df['W'] > 0]
print("Method 1 : storing the selected dataframe into another variable and accessing entire column X satisfying the selection condition : ")
print(resdf['X'])

print("Method 2 : Dirctly accessing : ")
print(df[df['W'] > 0].loc[['A', 'B', 'E'], ['X', 'Y']])

# combining conditions for selection from dataframe :

combres = df[(df['W'] > 0) & (df['Y'] > 0.5)]
# normal python and doesn't work on dataframes or series
print("AND operation : ")
print(combres)

combres = df[(df['W'] > 0) | (df['Y'] > 1)]
print("OR operation : ")
print(combres)

# to make current index into a column and make the new index the default numericals : (use inplace=True for permanent effect)
# df.reset_index()


# setting a new index:

newind = 'CA NY WY OR CO'.split()

df['States'] = newind
print("Using States as new index : ")
print(df.set_index('States'))  # for permanent use inplace
# using set_index removes the original index completely unless stored separately


""" 
DataFrame : 
          W         X         Y         Z
A  2.706850  0.628133  0.907969  0.503826
B  0.651118 -0.319318 -0.848077  0.605965
C -2.018168  0.740122  0.528813 -0.589001
D  0.188695 -0.758872 -0.933237  0.955057
E  0.190794  1.978757  2.605967  0.683509
Dataframe boolean for values greater than 0 :
       W      X      Y      Z
A   True   True   True   True
B   True  False  False   True
C  False   True   True  False
D   True  False  False   True
E   True   True   True   True
The Boolean Dataframe :
       W      X      Y      Z
A   True   True   True   True
B   True  False  False   True
C  False   True   True  False
D   True  False  False   True
E   True   True   True   True
Boolean dataframe to get conditionally selected values :
Method 1 :  using the booldf :
          W         X         Y         Z
A  2.706850  0.628133  0.907969  0.503826
B  0.651118       NaN       NaN  0.605965
C       NaN  0.740122  0.528813       NaN
D  0.188695       NaN       NaN  0.955057
E  0.190794  1.978757  2.605967  0.683509
Method 2 : Using the conditon directly :
          W         X         Y         Z
A  2.706850  0.628133  0.907969  0.503826
B  0.651118       NaN       NaN  0.605965
C       NaN  0.740122  0.528813       NaN
D  0.188695       NaN       NaN  0.955057
E  0.190794  1.978757  2.605967  0.683509
Rows where values in Column W are greater than 0 :
          W         X         Y         Z
A  2.706850  0.628133  0.907969  0.503826
B  0.651118 -0.319318 -0.848077  0.605965
D  0.188695 -0.758872 -0.933237  0.955057
E  0.190794  1.978757  2.605967  0.683509
Rows where values in Column Z are less than 0 :
          W         X         Y         Z
C -2.018168  0.740122  0.528813 -0.589001
Method 1 : storing the selected dataframe into another variable and accessing entire column X satisfying the selection condition :
A    0.628133
B   -0.319318
D   -0.758872
E    1.978757
Name: X, dtype: float64
Method 2 : Dirctly accessing :
          X         Y
A  0.628133  0.907969
B -0.319318 -0.848077
E  1.978757  2.605967
AND operation :
          W         X         Y         Z
A  2.706850  0.628133  0.907969  0.503826
E  0.190794  1.978757  2.605967  0.683509
OR operation :
          W         X         Y         Z
A  2.706850  0.628133  0.907969  0.503826
B  0.651118 -0.319318 -0.848077  0.605965
D  0.188695 -0.758872 -0.933237  0.955057
E  0.190794  1.978757  2.605967  0.683509
Using States as new index :
               W         X         Y         Z
States
CA      2.706850  0.628133  0.907969  0.503826
NY      0.651118 -0.319318 -0.848077  0.605965
WY     -2.018168  0.740122  0.528813 -0.589001
OR      0.188695 -0.758872 -0.933237  0.955057
CO      0.190794  1.978757  2.605967  0.683509
"""
