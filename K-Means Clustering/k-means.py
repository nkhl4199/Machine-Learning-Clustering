# K means clustering

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,3:].values

#Using the elbow method for optimal clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
        k_means=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
        k_means.fit(x)
        wcss.append(k_means.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number Of Clusters')
plt.ylabel('WCSS')
plt.show()

#Fitting the model
k_means=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y=k_means.fit_predict(x)

#Visualizing the result
plt.scatter(x[y==0,0],x[y==0,1],s=100,color='red',label='cluster1')
plt.scatter(x[y==1,0],x[y==1,1],s=100,color='blue',label='cluster2')
plt.scatter(x[y==2,0],x[y==2,1],s=100,color='green',label='cluster3')
plt.scatter(x[y==3,0],x[y==3,1],s=100,color='magenta',label='cluster4')
plt.scatter(x[y==4,0],x[y==4,1],s=100,color='cyan',label='cluster5')
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],s=300,color='yellow',label='centroid')
plt.title('Cluster of people')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()