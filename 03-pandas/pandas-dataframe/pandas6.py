import numpy as np
import pandas as pd

from numpy.random import randn

#groupby using pandas

data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}

df=pd.DataFrame(data)

print("Initial Dataframe : \n",df)

bycomp=df.groupby('Company')
print("Mean of Sales after Grouping by Company Column : \n",bycomp.mean())  #person column cannot be used for mean as it contains strings



