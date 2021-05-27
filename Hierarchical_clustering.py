import pandas as pd
from pandas import read_csv
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from string import digits
import scipy.cluster.hierarchy as shc

### download indians pima diabetes dataset ###

#read diabetes csv file
path = ''
df = pd.read_csv(path)
df=df.astype(float)


patient_data = df.iloc[:, 3:5].values
plt.figure()
plt.title("Patient Dendograms")
dend = shc.dendrogram(shc.linkage(df, method='ward'))

cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
cluster.fit_predict(patient_data)
plt.figure()
plt.scatter(patient_data[:,0], patient_data[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()