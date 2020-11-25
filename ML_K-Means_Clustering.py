
#Unsupervised Learning algo - works on unlabeled data
#Attemps to group similar clusters together in our data
    #Ex: clustering documents/customers together
#Allows us to cluster unlabeled data in an unsupervised ML algorithm


#How it works...
#Choose a number of clusters -> k
#Randomly assign each point to a cluster
#Until clusters stop changing, repeat the following:
    #For each cluster, computer the cluster centroid by taking the mean vector of points in the cluster
    #Assign each data point to the cluster for which the centroid is the closest


#Choosing a k-value:
    #Elbow method - compute the sum-of-squared error (SSE) for some values of k (for example 2,4,6,8)
    #The SSE is defined as the sum of the squared distance between each member of the cluster and its centroid
    #If you plot k against SSE, we will see that the error decreseases as k gets larger; this is because when the number of clusters increases, they should be smaller, so the distortion is also smaller
    #The idea of the elbow method is to choose the k at which the SEE decreases abruptly
    #This produces an 'elbow effect' in the graph,

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline (for jupyter ntbk)

from sklearn.datasets import make_blobs

data = make_blobs(n_samples=200, n_features=2, centers=4, 
                  cluster_std=1.8, random_state=101)

# print(data)
#print(data[0]) #is a 2-d array of x, y value pairs
#print(data[0].shape)
#print(data[1]) #is an array containing the cluster-values of where the x,y values belong

#plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='rainbow')

#data[0][:,0] this prints all rows in col 0 -- and -- data[0][:,1] prints all rows in col 1
#general representation --> table [ rowstart : rowend , colstart : col end ]

#plt.show()


#Using SciKit Learn to create a K-Means clustering algorithm
from sklearn.cluster import KMeans

#algo will start off by randomly assigning each observation to a cluster, and then find the centroid of each cluster
#it will iterate through 2 steps:
    #it reassigns datapoints to the cluster whose centroid is closest
    #it calculates the new centroid for each cluster
    #repeats that over and over until cluster-variation can't be reduced any further

# kmeans = KMeans(n_clusters=4) #we have to know the number of clusters before hand

# kmeans = KMeans(n_clusters=3) #trying with diff. number of clusters - 3
# kmeans = KMeans(n_clusters=2)   #trying with 2 number of clusters
# kmeans = KMeans(n_clusters=6)

kmeans.fit(data[0])

# print(kmeans.cluster_centers_)

# print(kmeans.labels_) #it will report back the labels it believes to be true for the clusters

#Creating a subplot of 1 row by 2 cols, sharey allows to share same axes
fig, (ax1, ax2) = plt.subplots(1,2,sharey=True, figsize=(10,6))

#Comparing actual labels with the labels of algo of k-means algo

#labels of the kmeans algorithm
ax1.set_title('K Means')
ax1.scatter(data[0][:,0], data[0][:,1], c=kmeans.labels_, cmap='rainbow')

#compared with the labels of the original data
ax2.set_title('Original')
ax2.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='rainbow')
plt.show()