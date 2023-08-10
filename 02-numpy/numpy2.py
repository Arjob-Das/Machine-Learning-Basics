import numpy as np
arr=np.arange(0,11)
print(arr)
#numpy doesnt create copies when slices of array or entire array is used to create a new variable
# arr.copy() must be used for this

a=[[1,2,3],[4,5,6],[7,8,9]]
print("List a : \n ",a)
print("List double bracket notation : \n ",a[1][2])
#print("List single bracket notation : \n ",a[1,2]) causes error

a=np.array(a)
print("Np array a : \n ",a)
print("NP array double bracket notation : \n ",a[1][2])
print("NP array single bracket notation : \n ",a[1,2])

#using slice on 2d matrices
print("Elements from rows 2 and 3 and columns 1 and 2 : \n",a[1:,:2])
arr_2d=np.arange(50).reshape(5,10)
print("Original array :\n",arr_2d)
print("Grabbing elements 13,14,23,24 : \n ",arr_2d[1:3,3:5])

arr=np.arange(1,11)
print("Original Array : \n",arr)
bool_arr=arr>5
print("Array by using boolean array to filter out original array : \n ",arr[bool_arr])
