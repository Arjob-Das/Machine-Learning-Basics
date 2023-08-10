import numpy as np
import pandas as pd

labels=['a','b','c']
my_data=[10,20,30]
arr=np.array(my_data)
d={'a':10,'b':20,'c':30}

print("Python List labels :\n ",labels)
print("Python List my data :\n ",my_data)
print("Numpy Array  labels :\n ",arr)
print("Python dictionary labels :\n ",d)

print(pd.Series(data=my_data))
x=pd.Series(data=my_data,index=labels)
print("Indexed Series :")
print(x)
x=pd.Series(d)
print("Series from dictionary : ")
print(x)

ser1=pd.Series([1,2,3,4],['USA','Germany','USSR','Japan'])
print(ser1)

ser2=pd.Series([1,2,5,4],['USA','Germany','Italy','Japan'])
print(ser2)

print("Value for index USA",ser1['USA'])

ser3=pd.Series(data=labels)
print(ser3)
print("Value for index 1",ser3[1])

print("Adding ser1 and ser2 : \n",ser1+ser2)

