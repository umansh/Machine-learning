import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import  make_blobs

#using make_blobs create sample data X and y
X,y = make_blobs(n_samples=1000 , centers=4 , cluster_std=0.60, random_state = 0)

#plot scallter
plt.scatter(X[:,0], X[:,1], s=15)
plt.show()

#use kmeans and fit X,y
kmeans = KMeans(n_clusters = 4)
kmeans.fit(X,y)
y_means = kmeans.predict(X)

#plot scatter plot and calculate centroid
plt.scatter(X[:,0], X[:,1],c=y_means, s=15)
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='red', s=100, alpha=0.9)
plt.show()

