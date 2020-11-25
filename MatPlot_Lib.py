
#MATPLOTLIB
#Most popular plotting library for Python
#Gives control over every aspect of a figure
#Desiged similar to MatLab's graphical plotting

import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline - jupiter show
#plt.show() - to show/print the plot
x = np.linspace(0,5,11)
y = x ** 2
print(x,'\n',y)
#two ways to create matplotlib plots - Functional and OOmethod

#-------------------------------------------------------------

#Functional 
plt.plot(x,y) 
#plt.plot(x,y,'r-') #color, linestyles etc
plt.xlabel('X label') 
plt.ylabel('Y label') 
plt.title('Title') 

plt.subplot(1,2,1) #plt.subplot(#ofRows, #ofCols, plotnumber)
plt.plot(x,y,'r')

plt.subplot(1,2,2)
plt.plot(y,x,'b')
#plt.show()


#-------------------------------------------------------------


#OO METHOD - better/formal way 

#Creating figure object
fig = plt.figure() #this figure-object can be thought of as an imaginary-blank-canvas
#Now adding axes to it
axes = fig.add_axes([0.1,0.1,0.4,0.4]) 
#list takes in 4 args - left_position, bottom_position, width, height (w.r.t the canvas)

#PLOTTING
axes.plot(x,y)
axes.set_xlabel('X Label')
axes.set_ylabel('Y Label')
axes.set_title('Title')
#plt.show()

fig = plt.figure() #Create a figure object
axes1 = fig.add_axes([0.1,0.1,0.8,0.8]) #add set of axes to it
axes2 = fig.add_axes([0.2,0.5,0.4,0.3])

axes1.plot(x,y)
axes2.plot(y,x)
#plt.show()

#------------------------------------------------------------------------

#Subplots -- Axes operations -- 

x = np.linspace(0,5,11)
y = x ** 2
 
#fig = plt.figure() 
#axes1 = fig.add_axes([0.1,0.1,0.8,0.8]) 
#axes1.plot(x,y)
#plt.show()

#SUBPLOTS - allows to specify n_rows and n_cols
fig, axes = plt.subplots(nrows=1,ncols=2) #prints two axes in one row, like:  |_  |_
#axes object is just a list of matplotlib axis, so you can do iterate/index operations through those axes

#for current_ax in axes:
 #   current_ax.plot(x,y)

axes[0].plot(x,y)
axes[1].plot(y,x)
axes[0].set_title('First Plot')

#Diff b/w plt.figure() and plt.subplots() is that
#plt.subplots is doing add_axes stuff for us automatically, based on the specified rows and cols
#axes.plot(x,y)


plt.tight_layout() #prevent overlapping in display
plt.show()


#-------------------------------------------------------------


#FIGURE SIZE -- ASPECT RATIO -- DPI 

fig = plt.figure() #dpi=100 ) 
#pass in figsize tuple (width, height) of figure in inches
#dpi = dots per inch, mostly use default
ax = fig.add_axes([0,0,1,1])

#LEGEND
ax.plot(x,x**2, label='X Squared')
ax.plot(x,x**3, label='X Cubed')

ax.legend() #Need to put this function here so it can be referenced by the args above

#ax.legend(loc = 0) #each location code corr.to. a location for the legend 
#to be placed somewhere, check documentation to find out which one is where

#ax.legend(loc=(0.1,0.1)) #can specify coordinates too, in case loc does not work

#ax.set_xlabel('x')

#plt.show()

#Similar w SUBPLOTS
fig,axes = plt.subplots(nrows=2, ncols=1, figsize=(8,2))
axes[0].plot(x,y)
axes[1].plot(y,x)
#plt.show()

#SAVING A FIGURE
fig.savefig('my_picture.png',dpi=200)

plt.tight_layout()
plt.show()


#-------------------------------------------------------------------


#SETTING COLORS

fig = plt.figure()
ax = fig.add_axes([0,0,1,1]) 
ax.plot(x,y, color='orange')
#ax.plot(x,y, color='FF8C00') #RGB HexCode for Color, FF8C00 is orange

#LINEWIDTH - lw
ax.plot(x,y, color='purple', linewidth=2, alpha=0.5)
#alpha is for opacity/transparency

#LINESTYLE - ls
ax.plot(x,y, color='purple', lw=2, linestyle ='-.')

#MARKER - marker, markersize, markerfacecolor, markeredgecolor, markeredgewidth
ax.plot(x,y, color='purple', lw=2, ls ='-.', marker='o', markersize=5,
markerfacecolor='yellow', markeredgewidth=3, markeredgecolor='green')
#marks points

#AXIS limitations
ax.set_xlim([0,1])
ax.set_ylim([0,2])

