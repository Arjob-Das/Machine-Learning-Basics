import pandas as pd
import numpy as np
from numpy.random import randn
df = pd.DataFrame({'col1': [1, 2, 3, 4],
                   'col2': [444, 555, 666, 444],
                   'col3': ['abc', 'def', 'ghi', 'xyz']})

print("Initial DataFrame : \n", df.head())

print("Unique values from colummn 2 of the DataFrame : \n",
      df['col2'].unique())  # shows numpy array

# showing the number of unique values instead of numpy arrays

print("Number Unique values from colummn 2 of the DataFrame using nunique : \n",
      df['col2'].nunique())

print("Number Unique values from colummn 2 of the DataFrame using len : \n",
      len(df['col2'].unique()))

print("Count for occurence of each value in column 2 : \n",
      df['col2'].value_counts())


def times2(x):
    return x * 2


print("Times 2 : \n", df['col1'].apply(times2))

print("Length of each element of column 3 : \n", df['col3'].apply(len))

print("Using lambda function : \n", df['col1'].apply(lambda x: x * 2))

print("List of Column Names : \n", df.columns)

print("Using df.index : \n", df.index)

# other methods of printing column names as a list instead of an object
print("Using df.columns.tolist() : \n", df.columns.tolist())
print("Using df.columns.values : \n", df.columns.values)
print("Using list(df) : \n", list(df))

print("Sort Values of column 2 and only show column 2 : \n",
      df['col2'].sort_values())
print("Change the entire dataframe after sorting by column 2: \n",
      df.sort_values(['col2']))

print("Boolean dataframe to show whether any value is null in dataframe df : \n",
      df.isnull())  # opposite of this function is df.notnull()

print("Sum of all values in dataframe that are not null : \n",
      df[df.notnull()])

data = {'A': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'],
        'B': ['one', 'one', 'two', 'two', 'one', 'one'],
        'C': ['x', 'y', 'x', 'y', 'x', 'y'],
        'D': [1, 3, 2, 5, 4, 1]}

df = pd.DataFrame(data)

print("New DataFrame : \n", df.head())

"""  To store a pivot table in another variable:
x = df.pivot_table(values='D', index=['A', 'B'], columns=['C']) 
x = pd.pivot(data=df, values=['D'], index=['A', 'B'], columns=['C'])
print(x) """

print("New Pivot Table : \n", df.pivot_table(
    values='D', index=['A', 'B'], columns=['C']))

""" Initial DataFrame : 
    col1  col2 col3
0     1   444  abc
1     2   555  def
2     3   666  ghi
3     4   444  xyz
Unique values from colummn 2 of the DataFrame : 
 [444 555 666]
Number Unique values from colummn 2 of the DataFrame using nunique :
 3
Number Unique values from colummn 2 of the DataFrame using len :
 3
Count for occurence of each value in column 2 :
 444    2
555    1
666    1
Name: col2, dtype: int64
Times 2 :
 0    2
1    4
2    6
3    8
Name: col1, dtype: int64
Length of each element of column 3 :
 0    3
1    3
2    3
3    3
Name: col3, dtype: int64
Using lambda function :
 0    2
1    4
2    6
3    8
Name: col1, dtype: int64
List of Column Names :
 Index(['col1', 'col2', 'col3'], dtype='object')
Using df.index :
 RangeIndex(start=0, stop=4, step=1)
Using df.columns.tolist() :
 ['col1', 'col2', 'col3']
Using df.columns.values :
 ['col1' 'col2' 'col3']
Using list(df) :
 ['col1', 'col2', 'col3']
Sort Values of column 2 and only show column 2 :
 0    444
3    444
1    555
2    666
Name: col2, dtype: int64
Change the entire dataframe after sorting by column 2:
    col1  col2 col3
0     1   444  abc
3     4   444  xyz
1     2   555  def
2     3   666  ghi
Boolean dataframe to show whether any value is null in dataframe df :
     col1   col2   col3
0  False  False  False
1  False  False  False
2  False  False  False
3  False  False  False
Sum of all values in dataframe that are not null :
    col1  col2 col3
0     1   444  abc
1     2   555  def
2     3   666  ghi
3     4   444  xyz
New DataFrame :
      A    B  C  D
0  foo  one  x  1
1  foo  one  y  3
2  foo  two  x  2
3  bar  two  y  5
4  bar  one  x  4
New Pivot Table :
 C          x    y
A   B
bar one  4.0  1.0
    two  NaN  5.0
foo one  1.0  3.0
    two  2.0  NaN
 """
