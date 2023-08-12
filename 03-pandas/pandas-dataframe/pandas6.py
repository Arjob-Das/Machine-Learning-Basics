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

""" 
Initial Dataframe : 
   Company   Person  Sales
0    GOOG      Sam    200
1    GOOG  Charlie    120
2    MSFT      Amy    340
3    MSFT  Vanessa    124
4      FB     Carl    243
5      FB    Sarah    350
6      FB   Andrew    300
f:\Self Study\ML Basics\ML-Basics\03-pandas\pandas-dataframe\pandas6.py:18: FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.
  print("Mean of Sales after Grouping by Company Column : \n", bycomp.mean())
Mean of Sales after Grouping by Company Column :
               Sales
Company
FB       297.666667
GOOG     160.000000
MSFT     232.000000
Sum of Sales after Grouping by Company Column :
          Sales
Company
FB         893
GOOG       320
MSFT       464
Standard Deviation of Sales after Grouping by Company Column :
               Sales
Company
FB        53.538148
GOOG      56.568542
MSFT     152.735065
Sum of Sales after Grouping by Company Column For FB :
 Sales    893
Name: FB, dtype: int64
Sum of Sales after Grouping by Company Column For FB :
 Sales    893
Name: FB, dtype: int64
Groupby Count function on Company Column :
          Person  Sales
Company
FB            3      3
GOOG          2      2
MSFT          2      2
Max function on Company Column : 
           Person  Sales
Company
FB         Sarah    350
GOOG         Sam    200
MSFT     Vanessa    340
Groupby along with description : 
         Sales
        count        mean         std    min    25%    50%    75%    max
Company
FB        3.0  297.666667   53.538148  243.0  271.5  300.0  325.0  350.0
GOOG      2.0  160.000000   56.568542  120.0  140.0  160.0  180.0  200.0
MSFT      2.0  232.000000  152.735065  124.0  178.0  232.0  286.0  340.0
Groupby along with description in different view : 
 Company              FB        GOOG        MSFT
Sales count    3.000000    2.000000    2.000000
      mean   297.666667  160.000000  232.000000
      std     53.538148   56.568542  152.735065
      min    243.000000  120.000000  124.000000
      25%    271.500000  140.000000  178.000000
      50%    300.000000  160.000000  232.000000
      75%    325.000000  180.000000  286.000000
      max    350.000000  200.000000  340.000000
Groupby along with description along with company selection : 
 Sales  count      3.000000
       mean     297.666667
       std       53.538148
       min      243.000000
       25%      271.500000
       50%      300.000000
       75%      325.000000
       max      350.000000
Name: FB, dtype: float64
"""
