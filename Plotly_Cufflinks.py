
#Plotly is an INTERACTIVE visualization-libary that is open-source
#Cufflinks connects plotly with pandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
#plt.show() to see graphs

from plotly import __version__
print(__version__) #Version must be greater than 1.9

import cufflinks as cf

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#Basically importing tools to work offline, without having to use paid feature of plotly data hosting

init_notebook_mode(connected=True) #connect notebook.ArithmeticError

cf.go_offline()

#==========================================

df = pd.DataFrame(np.random.randn(100,4), columns = 'A B C D'.split())
#dataframe with 100 rows and 4 cols. col-names are ['A','B','C','D'], just written fancily with .split()
#(Recall) randn is for data as a Normal Distribution
df.head()

df2 = pd.DataFrame({'Category':['A','B','C'], 'Values': [32,43,50]})
#created another datafram w cols Category, Values and the data

df.plot()
#By default, the .plot() makes a line plot - (aka time series economics chart)

#IPLOT - interactive line plot
df.iplot() #You can zoom, hover over for values, a lot of other useful options  

#SCATTER PLOT
df.iplot(kind='scatter',x='A',y='B', mode='markers')
#need to add mode to prevent default line connections

#BAR PLOT
df2.iplot(kind='bar', x='Category', y='Values')
 
#Using agg function for bar plot (if there is no categorical column)
df.sum().iplot(kind='bar')
df.count().ip lot(kind='bar')

#BOX PLOT
df.iplot(kind='box') 

#3D SURFACE PLOT
df3 = pd.DataFrame({'x':[1,2,3,4,5], 'y':[10,20,30,40,50], 'z':[100,200,300,400,500]})
#Need 3 variables for 3D plot
df3.iplot(kind='surface')


df31 = pd.DataFrame({'x':[1,2,3,4,5], 'y':[10,20,30,40,50], 'z':[5,4,3,2,1]})
df31.iplot(kind='surface', colorscale = 'rdylbu') #check out different color scales

df['A'].iplot(kind='hist', bins=25)

df.iplot(kind='hist', bins=25) #passing entire dataframe for overlapping to compare

#SPREAD - used a lot for stock data
df[['A','B']].iplot(kind='spread')
 
#BUBBLE PLOT - very similar to scatter plot except that it changes size of the points based on another variable
df.iplot(kind='bubble', x='A', y='B', size='C')
#used in metrics like Population, World GDP, Happiness Index, etc.

#SCATTER MATRIX PLOT - very similar to seaborn's PAIRPLOT (plots a graph for every pair-combination of cols in the dataframe)
#all data must be numeric to plot correctly
df.scatter_matrix()

#Github page for plotly has a lot of additional information/tools mentioned
#Also has a technical analysis(ta) page section for cufflinks - still in beta mode- but really cool/powerful for financial analysis
#Interesting to read and learn about the cufflinks ta page