import numpy as np
import pandas as pd
from numpy.random import randn
# Index Levels
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
print("Outside List : \n",outside)
print("Inside List : \n",inside)
print("Zipped tuples of outside with inside : \n",hier_index)
hier_index = pd.MultiIndex.from_tuples(hier_index)
print("Creating Multi Index of several levels using pandas : \n",hier_index)

df=pd.DataFrame(randn(6,2),hier_index,['A','B'])
 
print("Multi Index DataFrame : \n",df)
print("Row 1 of G1 of DataFrame : \n",df.loc['G1'].loc[1])

print("Unnamed DataFrame : \n",df)
df.index.names=['Groups','Num']
print("Named DataFrame : \n",df)

print("G2 of DataFrame : \n",df.xs('G2')) #or df.loc['G2'] can be used

print("Row 2 from G2 of DataFrame : \n",df.loc['G2'].loc[2])
print("B of Row 2 from G2 of DataFrame : \n",df.loc['G2'].loc[2]['B'])

print("A of Row 3 from G1 : \n",df.loc['G1'].loc[3]['A'])

#cross section of dataframe or series using xs() when there is a multilevel index

print("G1 from dataframe : \n",df.xs('G1'))

print("Named DataFrame : \n",df)
print("All values from dataframe where the inisde index i.e., Num =1 : \n",df.xs(1,level='Num'))

print("All values from dataframe where the inisde index i.e., Num =1 from G1 : \n",df.xs(('G1',1),level=('Groups','Num')))
