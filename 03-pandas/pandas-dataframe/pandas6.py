import numpy as np
import pandas as pd

from numpy.random import randn

# groupby using pandas

data = {'Company': ['GOOG', 'GOOG', 'MSFT', 'MSFT', 'FB', 'FB', 'FB'],
        'Person': ['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', 'Sarah', 'Andrew'],
        'Sales': [200, 120, 340, 124, 243, 350, 300]}

df = pd.DataFrame(data)

print("Initial Dataframe : \n", df)

bycomp = df.groupby('Company')
# person column cannot be used for mean as it contains strings
print("Mean of Sales after Grouping by Company Column : \n", bycomp.mean())
print("Sum of Sales after Grouping by Company Column : \n", bycomp.sum(
    numeric_only=True))  # person column cannot be used for mean as it contains strings
print("Standard Deviation of Sales after Grouping by Company Column : \n", bycomp.std(
    numeric_only=True))  # this used to be the default but has changed now
# person column cannot be used for mean as it contains strings

print("Sum of Sales after Grouping by Company Column For FB : \n", bycomp.sum(
    numeric_only=True).loc['FB'])  # person column cannot be used for mean as it contains strings

# repeating above usage of groupby without using an extra variable
print("Sum of Sales after Grouping by Company Column For FB : \n", df.groupby('Company').sum(
    numeric_only=True).loc['FB'])  # person column cannot be used for mean as it contains strings

print("Groupby Count function on Company Column : \n",
      df.groupby('Company').count())

print("Max function on Company Column : \n", df.groupby(
    'Company').max())  # strings are checked alphabetically
# the strings and numeric columns no longer keep their relation and only max for each company is shown irrespective of the fact whether the max in person has the max sales

print("Groupby along with description : \n", df.groupby('Company').describe())
print("Groupby along with description in different view : \n",
      df.groupby('Company').describe().transpose())

print("Groupby along with description along with company selection : \n", df.groupby(
    'Company').describe().loc['FB'])  # .loc is required if transpose is not used
# if transpose is used only ['FB'] is used without .loc
