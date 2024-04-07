import pandas as pd 
dataset = pd.read_csv('PotentialClients.csv').astype('float64')

import matplotlib.pyplot as plt 
plt.plot(dataset, 'x')
plt.title('Raw Potential Client Data')
plt.show()

#Empirical method for ideal number of clusters
import math
print("Empirical Method: \nNumber of Clusters: {0}".format(math.sqrt(dataset.shape[0]/2)))

# Elbow Plot method for KMeans model for 'k' in between 1 and 10
from scipy import cluster
initial = [cluster.vq.kmeans(dataset, i) for i in range(1, 10)]
plt.title('The Elbow Plot for Potential Clients')
plt.plot([var for (cent, var) in initial])
plt.show()

cent, var = initial[3]
#Use vq() to get as assignment for each observation
assignment, cdlist = cluster.vq.vq(dataset, cent)
plt.scatter(dataset.iloc[:,0], dataset.iloc[:,1], c=assignment)
plt.title("The Data Partitioned into 4 Clusters")
plt.show()