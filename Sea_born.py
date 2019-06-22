
#SeaBorn is a statistical plotting library
#Designed to work very well with pandas dataframe objects
#Has beautiful default styles
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#DISTRIBUTION PLOTS -- PLOTTING NUMERICAL DATA

tips = sns.load_dataset('tips') 
#tips is one of the built-in datasets in seaborn, just to practice with
print(tips.head())

# -- DISTPLOTS -- allows us to see a UniVariate observation, i.e. just-one-variable observation

sns.distplot(tips['total_bill'], bins=25, kde=False) 
#plt.show()
#bins is kinda like number of bars in the histogram
#Get a histogram and a KDE=Kernel-Density-Estimation--(the line curve along the histogram)
#kde=false removes the line-curve along the histogram



# -- JOINTPLOTS -- allows us to basically match up to distplots for BiVariate data, i.e. for comparing two variables

sns.jointplot(x='total_bill', y='tip', data=tips) 
#x is being plotted against y, so there two datasets x&y, from the tips table/dataframe
#default plot is scatter

#sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')  
#Hex plot is like scatter but in hexagon shape and darker at concentrated pts.
#More good looking of a plot

#sns.jointplot(x='total_bill', y='tip', data=tips, kind='reg') 
#reg is linear regression plot -- another visualization 

#sns.jointplot(x='total_bill', y='tip', data=tips, kind='kde')  
#kde is like contour, with darker at dense areas -- another visualization 
#plt.show()


#-- PAIRPLOT -- plot pair-wise relationships across an entire dataframe, for the numerical cols.
#Basically does Jointplot for every combination of column-pairs in the dataframe
#Also supports a color-hue-argument for CATEGORICAL(non-numerical) cols.

#sns.pairplot(tips, hue='sex', palette = 'coolwarm')
#hue is for coloring CATEGORICAL cols, palette is more of a color feature
#plt.show()
 

# -- RUG PLOTS -- 
sns.rugplot(tips['total_bill'])
plt.show()
#Rugplots show info like bar-code density. Dense lines at higher point, less-dense lines in thin areas

#------------------------------------------------------------------------------------------#

#CATEGORICAL PLOTS -- PLOTTING CATEGORICAL DATA

tips = sns.load_dataset('tips') 
print(tips.head())

# -- BAR PLOT -- general plot that allows to aggregate data based on some function (like mean, etc)
#sns.barplot(x='sex', y='total_bill', data=tips, estimator=np.std)
#estimator is the func. that we wish to perform, i.e. - avg, stdDev, sum, median etc.
#plt.show()


# -- COUNT PLOT -- same as bar plot but here estimator is explicitly counting the number of occurences
#Therefore, we need only one argument, y-axis is count
#sns.countplot(x='sex',data=tips)
#plt.show()

# -- BOX PLOT -- used to show distribution of categorical data
#also known as a Box and Whisker Plot
#sns.boxplot(x='day',y='total_bill',data=tips)
#See figure - Box plot shows the quartiles of the dataset and the whiskers show the rest of the distribution
#Can help us see outliers etc

sns.boxplot(x='day',y='total_bill',data=tips, hue='smoker')
#Hue feature - Plots w.r.t 'day' and as well as 'smoker' - Great feature of seaborn
#plt.show()

# -- VIOLIN PLOT -- v.similar to box plot, but it Gives more info about the distribution of plots itself
#harder to read though
#sns.violinplot(x='day',y='total_bill',data=tips)
sns.violinplot(x='day',y='total_bill',data=tips, hue='sex', split=True)
#plt.show()


# -- STRIP PLOT -- it will draw a scatterplot where one variable is categorical
#sns.stripplot(x='day', y='total_bill', data=tips, jitter=True, hue='sex', split=True)
#sns.stripplot(x='day', y='total_bill', data=tips, jitter=True)
#plt.show()


# -- SWARM PLOT -- same as strip plot except points are adjusted so they don't overlap
#combination of a strip plot and violin plot 
#=sns.swarmplot(x='day', y='total_bill', data=tips)
#plt.show()

#SIDE NOTE: can run both swarmplot and violinplot together to get a good pictorial representation
#sns.violinplot(x='day', y='total_bill', data=tips)
#sns.swarmplot(x='day', y='total_bill', data=tips)
#plt.show()


# -- FACTOR PLOT -- General plot method to plot any kind of plot
sns.factorplot(x='day', y='total_bill', data=tips, kind='bar')
#pass in data and choose the type-of-plot in kind to get what you want
#plt.show()


#--------------------------------------------------------------------------------------------------------------#

# -- MATRIX PLOTS -- HEATMAPS

# Heat-Maps - primary way of showing some sort of matrix plot
# Data must already be in matrix form for heatmap to work-
# -which means that the rows and col we take the data from must have a name.
#We Can do that through pivot table or by trying to get correlation data

#By correlation data
tc = tips.corr()
print(tc, '\n')

#HeatMap - colors in values based on some gradient scale
#Useful for seeing relation b/w data
#sns.heatmap(tc)
#sns.heatmap(tc, annot=True, cmap='coolwarm') 
#annot annotates the actual data on the graph
#plt.show()

fpt = flights.pivot_table(index='month',columns='year',values='passengers')
print(fpt)

sns.heatmap(fpt, cmap = 'magma', linecolor='white', linewidths=1)
plt.show()


# -- CLUSTER MAP -- 
#Going to use hierarchal clustering to produce a clustered version of  heat maps
#Good to group similar data values together in a heatmap
sns.clustermap(fp, cmap='coolwarm', standard_scale=1)


#--------------------------------------------------------------------------------------------------------------#

#GRIDS - Use seaborn's grids capability to automate subplots based off of features in our data

iris = sns.load_dataset('iris') 
print(iris.head(), '\n')

#GRIDS - Use seaborn's grids capability to automate subplots based off of features in our data
# g = sns.PairGrid(iris)
# #g.map(plt.scatter)
# g.map_diag(sns.distplot)
# g.map_upper(plt.scatter)
# g.map_lower(sns.kdeplot)
# plt.show()

tips = sns.load_dataset('tips')
print(tips.head())
g = sns.FacetGrid(data=tips, col='time', row='smoker') 
#Two Catgorical Columns time,row, #One numerical column to plot  
g.map(sns.distplot, 'total_bill')
plt.show()

g.map(plt.scatter, 'total_bill', 'tip')

#--------------------------------------------------------------------------------------------------------------#

#REGRESSION PLOTS - Use seaborn's grids capability to automate subplots based off of features in our data

# --LM plot - Linear Model plot - allows you to display linear models with seaborn

tips = sns.load_dataset('tips')
print(tips.head())

# sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', markers=['o','v'])#, scatter_kws={'s':100})
# #Customizable features 
# plt.show()

# sns.lmplot(x='total_bill', y='tip', data=tips, col='sex', row='time')
# plt.show()
 
sns.lmplot(x='total_bill', y='tip', data=tips, col='sex', hue='sex')#, aspect=0.6, size=8)
plt.show()


#--------------------------------------------------------------------------------------------------------------#


#STYLE and COLOR

tips = sns.load_dataset('tips')
print(tips.head())

# sns.set_style('ticks') #Background setting
#sns.set_style('white') #or whitegrid

# sns.despine() #Removes the top and right spines from graph

#plt.figure(figsize=(12,3)) #Resizing Figure

sns.countplot(x='sex',data=tips)
#plt.show()

# sns.set_context('poster', font_scale=2) #Font size
sns.set_context('notebook', font_scale=2)
plt.show()

#Palettes and Colors
sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', palette='seismic')