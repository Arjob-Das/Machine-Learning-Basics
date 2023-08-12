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


