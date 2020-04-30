
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans,DBSCAN 
from sklearn.cluster import AgglomerativeClustering as AC

file=open('./data/NLS/2D_points.txt')
l=file.readlines()
points=[]

for x in l:
    x1=float(x.split(" ")[0])
    x2=float(x.split(" ")[1])
    points.append([x1,x2])    

points=np.array(points)

plt.scatter(points[:,0],points[:,1])
plt.show()

kmean=KMeans(n_clusters=2,random_state=42).fit(points)

plt.scatter(points[:,0],points[:,1],c=kmean.labels_)
plt.show()

aclust=AC(n_clusters=2)
aclust.fit(points)

plt.scatter(points[:,0],points[:,1],c=aclust.labels_)
plt.show()

dbc=DBSCAN()
dbc.fit(points)

plt.scatter(points[:,0],points[:,1],c=dbc.labels_)
plt.show()

file.close()



file=open('./data/Ring/2D_points.txt')
l=file.readlines()
points=[]

for x in l:
    x1=float(x.split(" ")[0])
    x2=float(x.split(" ")[1])
    points.append([x1,x2])    

points=np.array(points)

plt.scatter(points[:,0],points[:,1])
plt.show()

kmean=KMeans(n_clusters=2,random_state=42).fit(points)

plt.scatter(points[:,0],points[:,1],c=kmean.labels_)
plt.show()

aclust=AC(n_clusters=2)
aclust.fit(points)

plt.scatter(points[:,0],points[:,1],c=aclust.labels_)
plt.show()

dbc=DBSCAN()
dbc.fit(points)

plt.scatter(points[:,0],points[:,1],c=dbc.labels_)
plt.show()

file.close()