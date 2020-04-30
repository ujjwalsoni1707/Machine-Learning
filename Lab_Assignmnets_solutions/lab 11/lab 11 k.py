import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN


file = open("2D_points.txt","r")

x = []
y = []

for line in file:
    a = line.split()
    x.append(float(a[0]))
    y.append(float(a[1]))
    
#print(x,y)

df = pd.DataFrame(list(zip(x, y)),columns=['x','y'])
print(df)

y_true = []

for i in range(500):
    y_true.append(0)
for i in range(250):
    y_true.append(1)

def purity_score(y_true, y_pred):
    contingency_matrix1=contingency_matrix(y_true,y_pred)
    print("contingency_matrix")
    print(contingency_matrix1)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix1)
    #print(row_ind,col_ind)
    #print(contingency_matrix1[row_ind,col_ind])
    print("Purity-score is:",end='')
    return (contingency_matrix1[row_ind,col_ind].sum())/(np.sum(contingency_matrix1))


kmeans = KMeans(n_clusters=2).fit(df)
centroids = kmeans.cluster_centers_
y_pred = kmeans.predict(df)

plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()

print(purity_score(y_true,y_pred))

clustering = AgglomerativeClustering().fit(df)
clustering = AgglomerativeClustering(affinity='euclidean',linkage='ward', n_clusters=2)
y_pred1=clustering.fit_predict(df)

plt.scatter(df['x'],df['y'], c=clustering.labels_, cmap='rainbow')
plt.show()
print(purity_score(y_true,y_pred1))

db = DBSCAN().fit(df)
y_pred2 = db.fit_predict(df)
plt.scatter(df['x'], df['y'],c=y_pred2, cmap='Paired')
plt.title("DBSCAN")
plt.show()
print(purity_score(y_true,y_pred2))

eps=float(input("enter eps : "))
min_sam=int(input("enter min samples : "))
db = DBSCAN(eps,min_sam).fit(df)
y_pred3 = db.fit_predict(df)
plt.scatter(df['x'], df['y'],c=y_pred3, cmap='Paired')
plt.title("DBSCAN")
plt.show()
print(purity_score(y_true,y_pred3))