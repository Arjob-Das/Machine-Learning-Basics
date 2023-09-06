import numpy as np
my_list = [1, 2, 3]
arr = np.array(my_list)
my_mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print("Normal List: \n", my_list)
print("My list as array : \n", arr)
print("Matrix i.e., nested list as multi dimensional array: \n ", np.array(my_mat))
x = np.arange(0, 10)
print("Np arrange: \n", x)
zero = np.zeros(3)
print("1D zeros array : \n", zero)
zero = np.zeros((2, 3))
print("2X3 array of 0 : \n", zero)
equal_interval = np.linspace(0, 5, 10)
print("0 to 5 10 intervals : \n", equal_interval)
equal_interval = np.linspace(0, 5, 100)
print("0 to 5 100 intervals : \n", equal_interval)
id_mat = np.eye(4)
print("4x4 identity matrix : \n ", id_mat)
x = np.random.rand(5)
print("np random rand 1D size 5 ", x)
x = np.random.rand(5, 5)
print("np random rand 2D size 5x5 ", x)
gaus_type = np.random.randn(2)
print("np random randn gauss type 1D size 2 ", gaus_type)
gaus_type = np.random.randn(4, 4)
print("np random randn gauss type 2D size 4x4 ", gaus_type)
random_integer = np.random.randint(1, 100)
print("1 random integer between 1 an 100 : \n ", random_integer)
random_integer = np.random.randint(1, 100, 10)
print("10 random integers between 1 an 100 : \n ", random_integer)
arr = np.arange(25)
print(arr)
ranarr = np.random.randint(0, 50, 10)
print(ranarr)
print("reshape arr : \n ", arr.reshape(5, 5))
print("max and min of the random array : \n",
      ranarr.min(), " and ", ranarr.max())
print("max index location : ", ranarr.argmax())
print("min index location : ", ranarr.argmin())
print("data type of array : ", arr.dtype)


""" 
Output :
Normal List: 
 [1, 2, 3]
My list as array : 
 [1 2 3]
Matrix i.e., nested list as multi dimensional array: 
  [[1 2 3]
 [4 5 6]
 [7 8 9]]
Np arrange:
 [0 1 2 3 4 5 6 7 8 9]
1D zeros array :
 [0. 0. 0.]
2X3 array of 0 :
 [[0. 0. 0.]
 [0. 0. 0.]]
0 to 5 10 intervals :
 [0.         0.55555556 1.11111111 1.66666667 2.22222222 2.77777778
 3.33333333 3.88888889 4.44444444 5.        ]
0 to 5 100 intervals :
 [0.         0.05050505 0.1010101  0.15151515 0.2020202  0.25252525
 0.3030303  0.35353535 0.4040404  0.45454545 0.50505051 0.55555556
 0.60606061 0.65656566 0.70707071 0.75757576 0.80808081 0.85858586
 0.90909091 0.95959596 1.01010101 1.06060606 1.11111111 1.16161616
 1.21212121 1.26262626 1.31313131 1.36363636 1.41414141 1.46464646
 1.51515152 1.56565657 1.61616162 1.66666667 1.71717172 1.76767677
 1.81818182 1.86868687 1.91919192 1.96969697 2.02020202 2.07070707
 2.12121212 2.17171717 2.22222222 2.27272727 2.32323232 2.37373737
 2.42424242 2.47474747 2.52525253 2.57575758 2.62626263 2.67676768
 2.72727273 2.77777778 2.82828283 2.87878788 2.92929293 2.97979798
 3.03030303 3.08080808 3.13131313 3.18181818 3.23232323 3.28282828
 3.33333333 3.38383838 3.43434343 3.48484848 3.53535354 3.58585859
 3.63636364 3.68686869 3.73737374 3.78787879 3.83838384 3.88888889
 3.93939394 3.98989899 4.04040404 4.09090909 4.14141414 4.19191919
 4.24242424 4.29292929 4.34343434 4.39393939 4.44444444 4.49494949
 4.54545455 4.5959596  4.64646465 4.6969697  4.74747475 4.7979798
 4.84848485 4.8989899  4.94949495 5.        ]
4x4 identity matrix :
  [[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
np random rand 1D size 5  [0.26842397 0.66705019 0.66463287 0.11922239 0.82638518]
np random rand 2D size 5x5  [[0.84557847 0.48623161 0.64091201 0.45355089 0.03618272]
 [0.80037592 0.22528182 0.27576467 0.1303663  0.37003466]
 [0.38089916 0.42684639 0.28815849 0.72964803 0.57172433]
 [0.74175418 0.14439579 0.96132225 0.51665018 0.23890911]
 [0.20608572 0.33216636 0.56742263 0.77562138 0.39007827]]
np random randn gauss type 1D size 2  [-0.26251345 -0.45718875]
np random randn gauss type 2D size 4x4  [[-0.64403544  0.00768096  0.08814746 -0.50169116]
 [ 1.04059806  0.71934899 -1.57109233  0.50052196]
 [-0.86542443 -0.81113854  0.52792391 -0.51194279]
 [ 0.09140723  0.5021364   1.02251138 -0.41219865]]
1 random integer between 1 an 100 :
  98
10 random integers between 1 an 100 :
  [18 69 43 98 19 99 65  7 17 73]
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24]
[ 7  3 23  6 25 12 30 17 13 39]
reshape arr :
  [[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]
 [20 21 22 23 24]]
max and min of the random array :
 3  and  39
max index location :  9
min index location :  1
data type of array :  int32

"""
#testing pycharm git
