
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#plt.show() to print graph

df1 = pd.read_csv('df1', index_col=0) #import csv file, set the first column as the index
df1.head()  #makes it a time series (?look into this)

df2 =  pd.read_csv('df2')
df2.head()

#Get a histogram for all the values in col 'A' in df1
df1['A'].hist(bins=30) #uses matplotlib-seaborn library directly
#Mostly uses histogram in pandas libary

#Making a plot
df1['A'].plot(kind='hist', bins=30) 
#After specifying the kind of plot, you can use the specific arguments related to that kind-of- plot
df1['A'].plot.hist() #Another way to make a hist

#Area Plot - Plots area with shaded regions
df2.plot.area(alpha=0.4) #passing in alpha
#alpha is color opacity/transparency

#Bar Plot
df2.plot.bar()
#Takes the index value as a category - so the index must be categorical
#Then it just plots the value of each column 
df2.plot.bar(stacked=True) #Makes it as a stack - multiple cols(w diff colors) in just one bar instead of each col having each bar

#Line Plot - time series graph - economics / charts
df1.plot.line(x=df1.index, y='B', figsize=(12,3), lw=1) #have to specify the x and y 
#Can add any matplotlib feature since these plots are built on top of mpl. Ex: markers, figsize, lw, ma

#Scatter Plot
df1.plot.scatter(x='A',y='B',c='C',cmap='coolwarm')
#plots 3D plot. plots A vs B in the plot and adds C with a different color in the same plot to differentiate

#With size s = 
df1.plot.scatter(x='A',y='B',s=df1['C']*100) 

#Box Plot
df2.plot.box()  

#For bivariate data - Hex plot
df = pd.DataFrame(np.random.randn(1000,2), columns=['a','b'])
df.head()
df.plot.hexbin(x='a',y='b',gridsize=25)

#KDE - Density plots
df2['a'].plot.kde()
df2['a'].plot.denisty()