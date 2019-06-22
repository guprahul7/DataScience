#PANDAS -> open src library built on top of NumPy
#aka by many people as Python's version of Excel, or R
#Allows fast data analysis, data cleaning and preparation
#Excels in performance and productivity
#Has built-in visualization features
#Can work with data from wide variety of srcs


#SERIES -> like arrays but they have axis labeling and are indexed by a label

import numpy as np
import pandas as pd 

labels = ['a','b','c'] 
myData = [10,20,30]
arr = np.array(myData) #cnvt myData list into array
d = {'a':10,'b':20,'c':30}

pd.Series(data=myData)
print(pd.Series(data=myData)) #automatic indexing with 0, 1, 2...

pd.Series(data=myData, index=labels)
print(pd.Series(data=myData, index=labels)) #data and labels together now

pd.Series(myData,labels) #is also okay, no need to always specify data= and index=

pd.Series(arr) #Can Serie-fy a numpy array as well.  
pd.Series(arr,labels) #same thing here

pd.Series(d) #converting dictionary d into a series
print(pd.Series(d)) #automatically makes keys as index and values as data

print(pd.Series(data=labels)) #series can hold any type of data object, here the data is labels, index is default 0,1,..

pd.Series(data = [sum, print, len]) 
print(pd.Series(data = [sum, print, len])) #can also hold built-in functions as the data

ser1 = pd.Series(data=[1,2,3,4],index=['USA','Germany','USSR','Japan'])
print(ser1)
ser2 = pd.Series(data=[1,2,5,4],index=['USA','Germany','Italy','Japan'])
print(ser2)
print(ser1['USA']) #prints 1, similar to a python dict

print(ser1+ser2) #here if there is a match between the two series it will execute(here add) the values, but if there is no match it will return NaN (null)


print('---------------------------------------------------------------------\n')



 #DATAFRAMES -> can be thought as 2-d series, so like a sheet

import numpy as np
import pandas as pd
from numpy.random import randn
np.random.seed(101)

#CREATING A dataFrame -> series of series
df  = pd.DataFrame(data=randn(5,4),index=['A','B','C','D','E'],columns=['W','X','Y','Z'])
print(df)

#Select a column 'W'
print(df['W']) #prints W column from the df DataFrame. 
#DataFrame can be thought of as a bunch of SERIES THAT SHARE THE SAME INDEX

#Select Multiple Columns
print(df[['W','Z']]) #For multiple columns, pass in a list of column names

#Creating new columns
df['new'] = df['W'] + df['Y']
print(df) #CREATING new column using operation on two existing columns

#Deleting a Column
df.drop('new',axis=1) 
#axis=0 is for index, axis=1 is for columns
#NOTE: you have to set inplace=True for original dataFrame to be modified
#The 'new' column will not be  deleted from the df DataFrame without inplace=True.

df.drop('new',axis=1,inplace=True) #This will delete 'new' from actual dataFrame
print(df)

df.drop('E',axis=0,inplace=True) #again inplace=True must be there for original DataFrame to be modified
print(df)

print(df.shape)

#Selecting ROWS 
print(df.loc['C']) #pass in label of index you want. #.loc is for location
#returns the contents of a row in a Series format, i.e. vertical table, same as if selecting a column
print(df.iloc[2]) #iloc is for numerical-based index

print(df.loc[['A','B']]) #returns rows in dataframe format


#Subset of rows and columns
print(df.loc['B','Y']) #returns value at row B, col Y -> single row single column
print(df.loc[['A','B'],['W','Y']]) #returns the subset of rows A,B and cols W,Y. 
#NOTE - for multiple rows/col we have to put list of rows/cols


print('----------------------------------------------------------------------\n')



#OPERATIONS on DataFrames

import numpy as np
import pandas as pd
from numpy.random import randn
np.random.seed(101) 
df  = pd.DataFrame(data=randn(5,4),index=['A','B','C','D','E'],columns=['W','X','Y','Z'])

#CONDITIONAL SELECTION 
print(df)

print(df[df>0]) #where value<0 it replaces it with NaN

print(df[df['W']>0]) #returns in dataframe/sheet format where the rows in the 'W' column are > 0

print(df[df['Z']<0]) #returns in dataframe format the rows where Z<0

#MULTIPLE indexing - for one column

print(df[df['W']>0]['X'])  #first filtering rows with'W'>0, then the 'X' col
#basically prints col 'X', but after filtering values where the 'W' values > 0 

#Simpler explanation
#new_df = df[df['W']>0] -> new_df = all rows whre 'W'>0
#print(new_df['X']) -> prints the 'X' col from new_df

print(df[df['W']>0][['Y','X']]) #list of columns
#multiple indexing to get corresponding columns

#MULTIPLE CONDITIONS
#Cannot use 'and' , we need to use & 
#For 'or', we need to use |
x = df[ (df['W']>0) & (df['Y']>1) ] 
print(x)

#RESET INDEX
rst=df.reset_index() #must put inplace=True for original df to be modified
#resets index to default, starting from 0,1,2...
#and makes the current index as a new column with name 'index'. Does not overwrite the old index
print(rst)

#SET INDEX

newIndex_list = 'CA NY WY OR CO'.split() #just making a list using split
df['States'] = newIndex_list #Making new column 'States' in df with values from the list newIndex
#OK because the dimensions of the-new-column and existing-ones-in-df match
print(df)

print(df.set_index('States')) #Must place inplace=True for permanent change
#sets values from the 'States' column to be the new index. 'States' is the name of the index-level
#overwrites old index so be careful. 
print(df)


print('---------------------------------------------------------------------\n')



#MULTI-INDEX and INDEX HIERARCHY

#Index Levels
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]

hier_index = list(zip(outside,inside)) #zip(list1,list2) -> pair
print(hier_index) #prints list of tuple-pairs ((G1,1),(G1,2)...(G2,3))

hier_index = pd.MultiIndex.from_tuples(hier_index)  
#function takes in a list of tuple pairs and creates a multi-index from it
print(hier_index)
print('\n') 

df = pd.DataFrame(data=randn(6,2),index=hier_index,columns=['A','B'])
print(df) #first-index G1 w subindex 1,2,3 || second index G2 with subindex 1,2,3
print('\n') 

print(df.loc['G1']) #returns G1 corresponding dataframe
print('\n') 

print(df.loc['G1'].loc[1]) #returns dataframe corresponding to index G1 and sub index 1
#If the row data is just 1-D, it returns as a series (column/vertical format), not as a row
print('\n') 

print(df.index.names) #prints none bc they don't have names yet
print('\n') 

df.index.names = ['Groups','Nums']
print(df)
print('\n') 

print(df.loc['G2'].loc[2]['B']) #prints value in index-G2 & subindex-2 (or row G2-2), and colB
#print(df.loc['G2'].loc[2,'B']) This works as well

#CROSS SECTION .xs
#For accessing inside-index levels or sub-indexes: indexes that are inside parent-indexes
x = df.xs(key=1,level='Nums') #key is the (sub)index (in this case it is #1) 
#level is the index-head-NAME or index-level-NAME under which the index 1 is located
print(x)


print('----------------------------------------------------------------------\n')



#MISSING DATA

#Creating DataFrame from Dictionary
df = pd.DataFrame({'A':[1,2,np.nan], 'B':[5,np.nan,np.nan],'C':[1,2,3]})
#default rows index = 0,1,2..., the dict keys correspond to columns and the values c.t.the data. Here the values are lists, so multiple data per column(key)
print(df), print('\n') 

#Dropping a data value
x = df.dropna() #default axis is 0, so drops all rows that have a cell with null value 
print(x), print('\n')

x=df.dropna(axis=1) #axis specified to 1, so drops all cols that contain a cell with null value
print(x), print('\n')

x=df.dropna(axis=0,thresh=2) #bc of thresh=2, if any row has at least 2 non-Na values, the row will not be deleted
print(x), print('\n')

#Filling data value
x = df.fillna(value='Fill value') #fills cells that have Na with 'Fill Value' 
print(x), print('\n')

#MAny times we want to fill the cell with other values, for example, the mean of the column
x = df['A'].fillna(value=df['A'].mean())
print(x)


print('--------------------------------------------------------------------\n')



#GROUPBY -> arranges into a smaller series/dataframe based on the column name
#GROUPBY -> allows you to group together rows based off of a column and perform some sort of function on them
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}

df = pd.DataFrame(data) #Making Dataframe to use
print(df), print('\n')

#Creating a GroupBy 
x=df.groupby('Company') #This will give us the groupby object, stored somewhere in memory
print(x),print('\n') #Returns <pandas.core.groupby.groupby.DataFrameGroupBy object at 0x10efd9390>
 

print(x.mean()) #returns in dataframe format with 'Company' as index and other columns as columns
print(x.sum())  #returns in dataframe format with 'Company' as index...sum
print('\n')

#can perform  normal dataFrame operations bc returns it in that format
print(x.sum().loc['FB']) #---#
print(x.sum().loc['FB'].shape) 
print('\n')

#Prints the sum of the 'FB' company
y = df.groupby('Company').sum().loc['FB'] #same as #---# above but all in one command
print(y)
print('\n')

#USEFUL FUNCTIONS of Groupby
print(x.min())  #print(x.max())
print('\n') 
print(x.count()) #returns count of each column 
print('\n')
print(x.describe()) #returns (in df format) all the useful statistics of the groupby,
print('\n')
print(x.describe().transpose())
print('\n')
print(x.describe().transpose()['FB'])
print('\n')



print('--------------------------------------------------------------------\n')



#MERGING, JOINING and CONCATENATING

df1 = pd.DataFrame({'A':['A0', 'A1', 'A2', 'A3'], 'B':['B0', 'B1', 'B2', 'B3'],
                    'C':['C0', 'C1', 'C2', 'C3'], 'D':['D0', 'D1', 'D2', 'D3']},
                    index=[0, 1, 2, 3])
print(df1), print('\n')

df2 = pd.DataFrame({'A':['A4', 'A5', 'A6', 'A7'],'B':['B4', 'B5', 'B6', 'B7'],
                    'C':['C4', 'C5', 'C6', 'C7'],'D':['D4', 'D5', 'D6', 'D7']},
                    index=[4, 5, 6, 7]) 
print(df2), print('\n')

df3 = pd.DataFrame({'A':['A8', 'A9', 'A10', 'A11'],'B':['B8', 'B9', 'B10', 'B11'],
                    'C':['C8', 'C9', 'C10', 'C11'],'D':['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])               
print(df3), print('\n')

#CONCATENATION -> basically GLUES together the dataframes
#by default it glues vertically (row-wise, bc default axis=0), if we set axis=1, it will glue col-wise
#NOTE: dimensions must match for concatenating
#NOTE: fills NaN value for empty/missing cells

#Concatenating Syntax - must pass in a list of dataframes that we want to concat-
ConCat = pd.concat([df1,df2,df3])
print(ConCat) #since axis=0 by default, it will concat- vertically, row-wise
print('\n')

concat_axis1 = pd.concat([df1,df2,df3],axis=1)
print(concat_axis1) #concat- col-wise bc axis=1 
print('\n')



#MERGE (similar to merging SQL tables together)
left_DF = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'], 
                        'key':['K0', 'K1', 'K2', 'K3']})
print(left_DF), print('\n')
   
right_DF = pd.DataFrame({'C':['C0', 'C1', 'C2', 'C3'],
                        'key':['K0', 'K1', 'K2', 'K3'],
                        'D':['D0', 'D1', 'D2', 'D3']})
print(right_DF), print('\n')

#Merging Syntax
merge = pd.merge(left_DF, right_DF, how='inner', on='key') #by default it will merge them on 'inner'
#'on' is where we specify the common col-name, if the values match, the merging will happen
print(merge), print('\n')
#Merge is better in the sense that it's more than just 'gluing' dataframes together


left_df = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                        'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})
print(left_df), print('\n')

right_df = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                         'key2': ['K0', 'K0', 'K0', 'K0'],
                            'C': ['C0', 'C1', 'C2', 'C3'],
                            'D': ['D0', 'D1', 'D2', 'D3'],})
print(right_df), print('\n')
 
#Merging with 2 common keys/col-names, default merge on how='inner'
merge_two = pd.merge(left_df, right_df, how='inner', on=['key1','key2'])
print(merge_two), print('\n')

#for all combinations of key1,key2 between the left_df and right_df, it will merge the corresponding values
#Can be understood as 'INTERSECTION Merge'

#Merge on 'outer' -> merges just like 'inner', but if there are no matching 
#values in the common keys, it will still merge and fill the empty cells as NaN
#can be understood as WHOLESOME MERGE
merge_two_outer = pd.merge(left_df, right_df, how='outer', on=['key1','key2'])
print(merge_two_outer), print('\n')

#merge on 'right' -> preserves right dataframe and merges w.r.t common col-name, merges with the left df. 
merge_two_right = pd.merge(left_df, right_df, how='right', on=['key1','key2'])
print(merge_two_right), print('\n')
#Can be understood as preference merge or selection merge

#merge on 'left' -> preserves the left dataframe and, w.r.t common col-name, merges with the right df. 
merge_two_left = pd.merge(left_df, right_df, how='left', on=['key1','key2'])
print(merge_two_left), print('\n')
#Can be understood as preference merge or selection merge


 #JOINING -> same as merging, but with index-keys instead of col-names
 #Joining is a convenient method for combining the columns of two potentially differently-indexed DataFrames into a single result DataFrame.
 #Like a merge, but the 'key' is on the index, instead of the col-name

left_dataframe = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                               'B': ['B0', 'B1', 'B2']},
                                index=['K0', 'K1', 'K2']) 
print(left_dataframe), print('\n')

right_dataframe = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                                'D': ['D0', 'D2', 'D3']},
                                index=['K0', 'K2', 'K3'])
print(right_dataframe), print('\n')

#Joining Syntax
join_RtoL = left_dataframe.join(right_dataframe)
print(join_RtoL), print('\n')
#joins the right_dataframe to the left_dataframe where it can find common index values
#If it can't find common index values, it just puts NaN

#joining on how='outer'
join_RtoL_outer = left_dataframe.join(right_dataframe,how='outer')
print(join_RtoL_outer) #joins both dataframes, merging where it can find common index-keys 


print('--------------------------------------------------------------------\n')



#ADVANCED OPERATIONS 

#PANDAS OEPRATIONS

df = pd.DataFrame({'col1':[1,2,3,4],
                   'col2':[444,555,666,444],  
                   'col3':['abc','def','ghi','xyz']})
print(df), print('\n')

#FINDING UNIQUE values and count-of-unique-values from columns
unique = df['col2'].unique() #returns array of unique values of col2
print(unique) 

N_unique = df['col2'].nunique() #Nunique returns count of unique values
N_uni = len(df['col2'].unique()) #another way: finding length of array containing unique values
print(N_unique), print(N_uni), print('\n')

#To get Unique Values and their count in one cmd.
unique_and_count = df['col2'].value_counts() #returns dataframe of unique values in one col. and their counts in another
print(unique_and_count), print('\n')

#Selecting Data 

#conditional selection
x = df[(df['col1']>2) & (df['col2']==444)] 
print(x),print('\n') #returns filtered dataframe based on condition

#APPLY method -> allows to implement custom function or built-in function
#Very useful with lambda function

#example with custom function
def times2(x):
    return x*2
df_apply = df['col1'].apply(times2)
print(df_apply)
print('\n'), print(df.apply(times2)), print('\n')

#similar example with lambda function
df_applyLambda = df['col1'].apply(lambda x: x*2)
print(df_applyLambda), print('\n')
 
#Remove column
col_drop = df.drop('col1',axis=1) #Must set Inplace=True for permanent change to df
print(col_drop) #returns same dataframe but without dropped col.
print('\n')

#Permanently Removing a Column*
#del df['col1']
#priny(df)

#Column and Row/Index info:
print(df.columns) #prints information about columns,  
print(df.index) #prints info about indexes
print('\n')

#SORTING and ORDERING a dataframe

sortDF = df.sort_values(by='col2') #sorts the dataframe w.r.t col2  
#inplace=False by default
print('\n')
 
#Finding Null values in a Dataframe:
nullDF = df.isnull() #will return boolean values, true/false, if null values exist/DNE
print(nullDF)

#Drop rows with NaN Values
dropnull = df.dropna()
print(dropnull), 

df = pd.DataFrame({'col1':[1,2,3,np.nan],
                   'col2':[np.nan,555,666,444],
                   'col3':['abc','def','ghi','xyz']})

#For null we use np.nan
#Filling null-values with something else
df_nullfill = df.fillna('FILL') #replaces null values with 'FILL'
print(df_nullfill), print('\n')

#PIVOT Table -> can create a dataframe with preferred choices of indexes, sub-indexes, columns, and values 
#advanced excel users 
data = {'A':['foo','foo','foo','bar','bar','bar'],
        'B':['one','one','two','two','one','one'],
        'C':['x','y','x','y','x','y'],
        'D':[1,3,2,5,4,1]}

df = pd.DataFrame(data)
print(df), print('\n')

pivot_table = df.pivot_table(values='D', index=['A','B'], columns=['C'])
print(pivot_table) #Where values don't exist for a particular intersection, NaN (null) value is filled


print('--------------------------------------------------------------------\n')



#DATA INPUT and OUTPUT - FILE READING and WRITING
#CSV - Excel - HTML - SQL

#NOTE: Files must be in same directory for python to read it using file name
#If file not in same directory, pass in whole file path, else only file name

#CSV
#Read CSV file syntax (pandas automatically assumes dataframe format)
#df_readCSV = pd.read_csv('filename.csv') 
#print(readCSV)

#Write to CSV file
#df.to_csv('MyFile', index=false)
#Index=False prevents automatic indexing while writing to a file
#So to keep the same data we have, we use index=false, it is imp.


#EXCEL
#pandas can read only data, not images/formula/macros from excel files

#Excel Read Syntax
df_readExcel = pd.read_excel('Excel_Sample.xlsx', sheet_name='Sheet1')
print(df_readExcel), print('\n')
df = pd.read_excel('Excel_Sample.xlsx', sheet_name='Sheet1')

#Excel Write Syntax
#df.to_excel('Excel_Sample.xlsx',sheet_name='Sheet1')


#HTML
#readHTML = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')
#print(type(readHTML)), print('\n')

#Pandas tries to find every table-element in the HTML file - (by finding table references in the backend HTML code (inspect-element or view-page-source))
#When pandas finds relevant table-elements, it makes a list of them and converts each item in that list to a dataframe
#So we have to cycle through that list until we find what we were looking for.
#In this case it should be the first item,readHTML[0], bc there is only one table-element in the HTML file
#readHTML[0] is a DATAFRAME

#print(readHTML[0]), print('\n')
#print(readHTML[0].head()),  print('\n')
#Some things don't copy over exactly, so null values are filled, pandas tries it's best


#SQL
#Pandas isn't probably the best way to read a sql database bc there are 
#   many flavors of sql engines such as postscripts, sql, MySQL, sqlLite etc.
#But we're gonna build a very simple sql engine that will be temporarily held in memory
#   in order to see how we can use Pandas to read tables completely as DataFrames

#Recommended to do a search for a spcfic driver dpndent on your spcfic sql engine
#ForExmpl -> in ntbk, if you're using Postgres sql, you should use a lib called Pyscho PGE 2 which is 
#   specifically designed to work with Postgres sql
#If you're using MySql, you should use pi MySql
#NOTE: Look for specialized python libraries for whatever version of sql you're working with

from sqlalchemy import create_engine
#This will allow us to create a very simple sql engine in memory

engine = create_engine('sqlite:///:memory:') 
#This code has created a very temporary, very small, SQLite-engine-database that's running in memory
print(df), print('\n')

#writing to the temp-engine running in memory
df.to_sql('my_table', engine,index=False) #this engine is a connection, usually
df_readSQL = pd.read_sql('my_table', con=engine) #con is which connection we want
print(df_readSQL), print('\n')

#example with index=true (by default), we get unncessesary automatic indexing
df.to_sql('my_table2', engine)
df_readSQL = pd.read_sql('my_table2', con=engine)
print(df_readSQL)

print('--------------------------------------------------------------------\n')
