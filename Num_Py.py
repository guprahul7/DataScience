#NumPy is a Linear Algebra Library for Python. Almost all of PyData EcoSystem rely on NumPy
#NumPy has binding to C libraries
#Main feature - NumPy Arrays -> broken to vectors and matrices
#Vectors are strictly 1-D arrays 
#Matrices are 2-D, but note: Matrices can still have only 1 row or 1 column

#Casting python lists into array (1-D) or matrix (technically array with 2-D)

import numpy as np

my_list = [1,2,3] 
print(my_list)
arr = np.array(my_list) #makes list into array (i.e. a row vector, so without commas, lists hv commas)
print(arr[0])
print(arr)
print("\n")

my_list = [[1,2,4],[5,62,7],[55,67,73]]  #list of lists
arr = np.array(my_list) #converted into a matrix
print(arr)
print("\n")

print(np.arange(0,10,2))  #np.aRange(start,stop,step) to obtain Numbers in the specified range
print(np.arange(20)) #default start=0, default step=1, prints 0 -> 19 
print("\n")

print(np.zeros(3))  #np.zeros() to print row with 3 zeros
print(np.zeros((2,5)))  #np.zeros(tuple), prints zeros in 2D
print("\n")

print(np.ones(5)) #prints row with 5 ones
print(np.ones((2,7)))
print("\n")

print(np.linspace(0,5,5)) #prints 5 pts between 0 and 4 -- LINSPACE = DIVISIONS
#np.linspace(start,stop, count in b/w) In linspace we hv two points and the count of points reqd. b/w those two points
print("\n")

print(np.arange(0,5,5)) #ARANGE = INCREMENTS, hence, prints 1 pt bc it tries to go from 0 to 4 in increments of 5
#in np.arange we have two points and step size.
print("\n")

print(np.eye(3)) #Identity matrix with 3 rows x 3 colu
print("\n")

print(np.random.rand(5)) #1-d array of Five Random numbers (default b/w 0 to 1) from a uniform distribution.
print(np.random.rand(5,3)) #For 2-D, pass two arguments
print("\n")

print(np.random.randn(3)) #Rand_N is for numbers from normal distribution 
print(np.random.randn(5,3)) #for 2-d
print("\n")

print(np.random.randint(0,100,3)) #rand_int(start,end, count) is for random integers, here 3 random numbers b/w 0 and 99
print(np.random.randint(100)) #default start=0 and count=1 #prints 1 number b/w 0 and 99 
print("\n")

#RESHAPE
arr = np.arange(25) #array with 25 numbers b/w 0 and 24 
print(arr)
ranArr = np.random.randint(0,50,10) #array with 10 random integers b/w 0 and 49
print(ranArr)
print(arr.reshape(5,5)) #reshapes arr from 1-d to 2-d, but must have same count of elements for no error
print("\n")

print(ranArr.max()) 
print(ranArr.argmax()) #argmax() returns the index of the max
print(ranArr.min())
print(ranArr.argmin()) #argmax() returns the index of the min
print("\n")

print(arr.shape) #arr.shape returns the size of the matrix/vector, in this case 5 x 5
arr = arr.reshape(5,5)
print(arr.shape)
print("\n")

print(arr.dtype) #arr.dtype returns the Data-Type of the array

from numpy.random import randint #saves time by not having to write np.random.randint() all the time
print(randint(10)) #prints 1 number b/w 0 and 9
print('-------------------------------------------------------------------------------------\n')


#-----------------------------------------------------------------------------------------------------------------------------------------#
#NumPy Array indexing and selection
#FOLLOWS normal indexing as lists, so x[2], x[1:], x[4:9] etc etc.

#BROADCASTING -> modifies original b/c it does not make a copy, it is only a 'viewer'
#__Grabbing a slice of array is only a 'view' to array, not a copy. Modifying slice will modify original 

import numpy as np

arr = np.arange(0,11)
print(arr) 
arr[0:5] = 100 # changes values of first five elements to 100
print(arr)
print("\n")

arr = np.arange(0,11)
print(arr) #prints 0 -> 10
slice_of_arr = arr[0:5] #slice is first 5 elements of arr; 0,1,2,3,4
slice_of_arr[:] = 99  #all elements of slice are now 100 
print(slice_of_arr) #prints 100,100,100,100,100
print(arr) #Prints 100, , , ,100,6,7,8,9,10. Ideally arr should print 0 -> 10 but since slice is NOT A COPY and is declared as part of arr, when slice is modified, so is arr.
print("\n")
#NUMPY avoids copying arrays for memory sake, 

#TO make a COPY, use .copy()
arr = np.arange(0,11)
arrCopy = arr.copy()
arrCopy[:] = 88
print(arr)
print(arrCopy) 
print("\n")

arr2D =  np.array([[5,10,15],[20,25,30],[35,40,45]])
print(arr2D)
print(arr2D[1][1]) #indexing is array[row][col], for entire row/col, only index with row
#BETTER WAY 
print(arr2D[1,1]) #array[row,col]
#GRABBING Sub-MATRICES
print(arr2D[:2,1:]) #same slice syntax, grab all rows until row2 and all col after column1, including col1
print("\n")

#Array Condition
arr = np.arange(0,11)
print(arr)
print(arr[arr<3]) #prints arr but with condition applied, only values <3 
#Explanation Broken Down
bool_arr = arr>5 #array with values > 5 from arr
print(bool_arr) #returns array with True-False depending on condition

print(arr[bool_arr]) #returns values in arr that corresponde to True in bool_arr, in this case 6,7,8,9,10

print('-------------------------------------------------------------------------------------\n')


#-----------------------------------------------------------------------------------------------------------------------------------------#
#NumPy Operations - Array w Array ---- Array with Scalars ---- Universal Array Operations

import numpy as np

arr = np.arange(1,11)
#ARITHMETIC
print(arr+arr) #adds element by element
#print(arr-arr)
#print(arr*arr)

#Scalar
print(arr + 100) #adds 100 to each element

#NOTE - numpy will give warnings, not errors for illogical math operations 

#UNIVERSAL operations 
print(np.sqrt(arr)) #returns sqrt
np.exp(arr)#e^x
np.max(arr) #same as arr.max()
np.sin(arr)
np.log(arr)
np.sum(arr) #returns sum of the matrix/array
#or mat.sum()
np.std(arr) #returns Std.Deviation
#or mat.std()

mat = np.arange(1,26).reshape(5,5)
mat.sum(axis=0) #sum of all columns in an arr/mat
mat.sum(axis=1) #sum of all rows in an arr/mat