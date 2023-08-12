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

""" 
Data Frame : 
           W         X         Y         Z
A  2.706850  0.628133  0.907969  0.503826
B  0.651118 -0.319318 -0.848077  0.605965
C -2.018168  0.740122  0.528813 -0.589001
D  0.188695 -0.758872 -0.933237  0.955057
E  0.190794  1.978757  2.605967  0.683509
W column in Data Frame :
 A    2.706850
B    0.651118
C   -2.018168
D    0.188695
E    0.190794
Name: W, dtype: float64
A    2.706850
B    0.651118
C   -2.018168
D    0.188695
E    0.190794
Name: W, dtype: float64
Multiple Columns :
           W         Z
A  2.706850  0.503826
B  0.651118  0.605965
C -2.018168 -0.589001
D  0.188695  0.955057
E  0.190794  0.683509
Modified DataFrame :
           W         X         Y         Z       new
A  2.706850  0.628133  0.907969  0.503826  3.614819
B  0.651118 -0.319318 -0.848077  0.605965 -0.196959
C -2.018168  0.740122  0.528813 -0.589001 -1.489355
D  0.188695 -0.758872 -0.933237  0.955057 -0.744542
E  0.190794  1.978757  2.605967  0.683509  2.796762
Displaying temorarily dropped new column :
           W         X         Y         Z
A  2.706850  0.628133  0.907969  0.503826
B  0.651118 -0.319318 -0.848077  0.605965
C -2.018168  0.740122  0.528813 -0.589001
D  0.188695 -0.758872 -0.933237  0.955057
E  0.190794  1.978757  2.605967  0.683509
Showing that df reamins unchanged by previous action :
           W         X         Y         Z       new
A  2.706850  0.628133  0.907969  0.503826  3.614819
B  0.651118 -0.319318 -0.848077  0.605965 -0.196959
C -2.018168  0.740122  0.528813 -0.589001 -1.489355
D  0.188695 -0.758872 -0.933237  0.955057 -0.744542
E  0.190794  1.978757  2.605967  0.683509  2.796762
After dropping df permanently :
           W         X         Y         Z
A  2.706850  0.628133  0.907969  0.503826
B  0.651118 -0.319318 -0.848077  0.605965
C -2.018168  0.740122  0.528813 -0.589001
D  0.188695 -0.758872 -0.933237  0.955057
E  0.190794  1.978757  2.605967  0.683509
Displaying permanently dropped E row :
           W         X         Y         Z
A  2.706850  0.628133  0.907969  0.503826
B  0.651118 -0.319318 -0.848077  0.605965
C -2.018168  0.740122  0.528813 -0.589001
D  0.188695 -0.758872 -0.933237  0.955057
Row A :
W    2.706850
X    0.628133
Y    0.907969
Z    0.503826
Name: A, dtype: float64
Accessing Row C using numerical index position :
W   -2.018168
X    0.740122
Y    0.528813
Z   -0.589001
Name: C, dtype: float64
Data from Row B Column Y :
-0.8480769834036315
Data from Rows A,B Columns W,Y :
          W         Y
A  2.706850  0.907969
B  0.651118 -0.848077
DataFrame :
          W         X         Y         Z
A  0.302665  1.693723 -1.706086 -1.159119
B -0.134841  0.390528  0.166905  0.184502
C  0.807706  0.072960  0.638787  0.329646
D -0.497104 -0.754070 -0.943406  0.484752
E -0.116773  1.901755  0.238127  1.996652

"""
