

#a = [2,3,4]
#a.append(5)
#a.append(6)
#print(a) 

#using .format to print 
xx = 14
yy=20
print ('my numbers are: {} {}'.format(xx,yy))
num = 12
name = 'Sam'
print('My number is: {one}, and my name is: {two}'.format(one=num,two=name))
#slice operations s[0:3] or s[0:]


#nested Lists
a = [1,2,3,[4,5,[6,7]]]
print(len(a[3][2]))  #len(list) is used to find the lenght of a list
#Tuples cannot be altered, they are immutable
tup1=(33,49)
#Sets can be altered but they have only unique elements
set1 = {40,50,70}
set1.add(4)
print(set1)


#Dictionary always has a key and value pair, they don't have any order, it is a hash map with distinct key-value pairs
var1 = 9
d = {'key1':'item1','key2':var1} #string values need quotes, numericals and variables don't
print(d['key2']) #prints var1 which is 9 

#You can have nested dictionaries and nested lists, easy
d = {'x1':{'innerX1':[32,44]}}
print(d['x1']['innerX1'][0]) #prints 32

d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}
a = d['k1'][3]['tricky'][3]['target'][3]
print(a) #prints hello

#Important dictionary methods - .keys(), .values(), .items()
newDict = {'k1': 'value1', 'k2': 'value2'}
print(newDict.keys()) #.keys() prints out the keys, so k1 and k2
print(newDict.values())#prints out values
print(newDict.items()) #.items() prints out  contents of dictionary, so key,value pairs in a list, but w/o order
 #Imp. to remember that they don't have any order

#Comparison Operators == , <= , >= etc 

#FOR LOOP 
#RANGE() in range the end index is not included, so this prints 9 times from 1 to 9.
for sth in range(1,10):
    print('hello')


#List Comprehension 
#using for/while to modify loops quickly 
x = [1,2,3,4]
y = [item**2 for item in x] ##
print(y)

out = []
for item in x:
    out.append(item**2)
print(out)



#FUNCTIONS
def funct(n=3):
    print(n)
funct() 
#for running functions w no arguments, eg. to print sth, assign the printing_value to argument in the parenthesis
#this function does not return anything, just prints when called



def square(n): #assign value to a function that returns a value
    """
    Triple quotes is for commenting inside a func. to describe what the func. is used for
    """
    n = n**2
    print(n)
    return (n) #this returns a value, so assign this func. to sth like this x = square(n)

    

#LAMBDA and MAP
#So lambda is used as a shortcut sbstut for a function that returns some value.
#It is used to quickly sbstute the function parameter and get the return value. 
#Format is lambda num: num*2 //here var is the supposed parameter of a function and var*2 is the returned value
#Example

def times2(num):
    return num*2
print('testing lambda')
print(times2(39))

z = lambda sm_var: sm_var*2
print(z(31))

#MAP 
#map is used to map/copy from an iterable object using a function/lambda

seq = [10,11,12,13,14]

map(times2,seq) #this maps the contents of seq with the function and stores it smwhere in memory
old_list = list(map(times2,seq)) #this will give us [20,22,24,26,28]. seq is mapped w function times2

#This is where lambda is helpful. Instead of creating a new function times2, we just used lambda for quickness
new_list = list(map(lambda num:num*2, seq))
print(new_list)

#Filter
seq = [10,11,12,13,14]
filter(lambda item: item%2 == 0,seq)
fil = list(filter(lambda item: item%2 == 0,seq))
print(fil)

seq = ['soup','dog','salad','cat','great']
newlist = list(filter(lambda text: text[0]=='s',seq)) #filter texts beginning with s
print(newlist) #prints soup, salad in a list



#IMPORTANT METHODS
#.lower() -> lowercases the string
#.upper() -> uppercase all values
#.append() -> add item to list
#.index(x) -> returns the index of x from a list/iterable
# .count(x) -> returns the count of x from a list/iterable
s = 'My name is Sam 1000'
print(s.lower()) 

#SPLIT -> by default it splits at whitespace and puts the contents in a list, 
print(s.split()) #all the contents, including numbers, are str values
s = 'My#name # is# xyz'
print(s.split('#')) #splits at delimiter # and puts contents in a list
print(s.split('#')[1]) #index is used to get contents of the list, since split puts things in a list
print(s.split('#')[1:]) #use of slice operation

#POP -> pops the last item in a list (unless index is specified), and that item can be assigned to a variable
lst = [11,22,33,44,55]
a = lst.pop()
print(a)
b = lst.pop(2) #pops 3rd item in the list
print(b)

#IN statement -> Check if something in a List or String
s = 'My name is Sam'
x = 'Sam' in s
print(1000 + x) #should prints 1001
x = 'sam' in s
print(1000+x) #should print 1000

lst = [11,22,33,44,55]
zz = 'x' in lst #returns false
print(zz)
if (zz==0):
    print('False=0') 

lst = [11,22,33,44,'x']
zz = 'x' in lst #returns True
print(zz)
if (zz==1):
    print('True=1')


#TUPLE Unpacking
#A lot of functions return values in this form, tuples in a list, so it is imp. to know tupleUnpacking
x = [(12,13),(14,15),(16,17)]
print(x[1]) #prints 14,15
print(x[1][0]) #prints 14

for a in x:
    print(a) #prints out all the tuple pairs

for (a,b) in x:
    print(a) #prints 1st values of the tuples

#ZIP is used to "PACK" tuple-pair values from two different lists
listA = ['A','B','C','D','E','F']
list1 = [1,2,3,4,5,6]
a = list(zip(listA,list1)) #creates a list of (tuple, pairs) from corresponding values of list_1 and list_2
print(a) #prints((A,1),(B,2),(C,3)...)
