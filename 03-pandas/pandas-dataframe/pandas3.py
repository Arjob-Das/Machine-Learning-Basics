import numpy as np
import pandas as pd

from numpy.random import randn
np.random.seed(101)

df=pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])

print("DataFrame : ")
print(df)

#conditional selection

print("Dataframe boolean for values greater than 0 : ")
print(df>0)

booldf=df >0

print("The Boolean Dataframe : ")
print(booldf)

#combining boolean dataframe to get conditionally selected values 

print("Boolean dataframe to get conditionally selected values : ")
print("Method 1 :  using the booldf : ")
print(df[booldf])
print("Method 2 : Using the conditon directly : ")
print(df[df>0])

#using above on subsets
print("Rows where values in Column W are greater than 0 : ")
print(df[df['W']>0])

print("Rows where values in Column Z are less than 0 : ")
print(df[df['Z']<0])

#selecting specific results from conditionally selected dataframe:

resdf=df[df['W']>0]
print("Method 1 : storing the selected dataframe into another variable and accessing entire column X satisfying the selection condition : ")
print(resdf['X'])

print("Method 2 : Dirctly accessing : ")
print(df[df['W']>0].loc[['A','B','E'],['X','Y']])

#combining conditions for selection from dataframe : 

combres=df[(df['W']>0) & (df['Y']>0.5)]  
# normal python and doesn't work on dataframes or series
print("AND operation : ")
print(combres)

combres=df[(df['W']>0) | (df['Y']>1)]  
print("OR operation : ")
print(combres)

#to make current index into a column and make the new index the default numericals : (use inplace=True for permanent effect)
#df.reset_index()


#setting a new index:

newind='CA NY WY OR CO'.split()

df['States']=newind
print("Using States as new index : ")
print(df.set_index('States') ) #for permanent use inplace
#using set_index removes the original index completely unless stored separately


