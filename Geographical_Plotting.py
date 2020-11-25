
#By its nature, Geographical Plotting is very challenging, due to the various formats the data can come in
#A lot of times you have to find a specific library to work with for your specific data source
#For this course, we're going to work with Plotly
#Note: Matplotlib also has a 'BaseMap' extension, - which allows to create static geographical plots using Matplotlib

import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go #to use go.figure

#basic charts/syntax cheat sheet in notebook, since it can get challenging to use plotly for these graphs



#--- CHOROPLETH Maps - USA ---- 

#First thing is buiding our data dictionary
#For some reason the plotly library likes using a dictionary casting call instead of building it out as a dictionary 

#Data must be given in a dict casting way
data = dict( type='choropleth', 
            locations = ['AZ','CA','NY'], 
            locationmode = 'USA-states', 
            colorscale = 'Portland', 
            text = ['Arizona', 'Cali', 'New York'], 
            z = [1.0,2.0,3.0], 
            colorbar = {'title': 'Colorbar Title Goes Here'})

#type= what type of geographical plot we are doing
#locations= array or list of values that will be represented as the subject of z (z is the actual data with coloring, but what z kind-of represents is the location)
#locationmode= there are various locationmodes we can provide, can look it up
#colorscale= we can also provide various colorscales, here we are using portland colorscale
#text= this is an array/list of the text that hovers over each of the locations
#z = these are the actual values that are going to be shown in a color scale

layout = dict(geo={'scope':'usa'})
#this is a nested dictionary object, (notice above)

import plotly.graph_objs as go

choromap = go.Figure(data = [data], layout = layout)
#go.Figure(data, layout) is also fine

iplot(choromap)
#can play around with different colorscales ('Jet','Greens') etc

plot(choromap) #opens a new file as an html and we can save the file. It's not interactive


#=============================

#Real File/Data Example

df = pd.read_csv('2011_US_AGRI_Exports')
df.head()

data = dict(type = 'choropleth',
            colorscale = 'YIOrRd',
            locations = df['code'],
            locationmode = 'USA-states',
            z = df['total exports'],
            text = df['text'],
            marker = dict( line = dict(color= 'rgb(255,255,255)',width=2)), #marker is to add lines for visuals
            colorbar = {'title': 'Millions USD'}
            )

layout = dict(title = '2011 US Agriculture Exports by State',
            geo = dict(scope='usa', showlakes=True, lakecolor='rgb(85,173,240)')
              )

choromap2 = go.Figure(data = [data], layout = layout)



#==========================================================================================

#CHOROPLETH Maps - World

df = pd.read_csv('2014_World_GDP')
df.head()

data = dict( type = 'choropleth',
             locations = df['CODE'],
             z = df['GDP (BILLIONS)'],
             text = df['COUNTRY'],
             colorbar = {'title': 'GDP in Billions USD'}
            )

layout = dict( title = '2014 Global GDP',
                geo = dict(showframe = False, projection = {'type':'Mercator'})
             )      #type: natural earth is a nice one to look at
                    #look up documentation to find all the different options
choromap3 = go.Figure(data=[data], layout = layout)
iplot(choromap3)