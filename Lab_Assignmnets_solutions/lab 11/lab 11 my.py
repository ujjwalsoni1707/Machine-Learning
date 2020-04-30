# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:30:03 2019

@author: Ujjwal Soni
"""

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import gaussian_mixture
from sklearn import metrics
import numpy as np
#from yellowbrick.cluster import KElbowVisualizer
def purity_score(actual,predicted):
    mat = metrics.cluster.contingency_matrix(actual,predicted)
    a = np.sum(np.amax(mat,axis =0))/np.sum(mat)
    return a
    
def PCAA(dataframe):
    pca = PCA(n_components=2)
    newdata = pd.DataFrame(pca.fit_transform(dataframe))
    return newdata


def kmen(x,i,df):
    kmeans = KMeans(n_clusters = i,random_state = 0).fit(x)
    pred = kmeans.labels_
    centres = kmeans.cluster_centers_
    y_kmeans = kmeans.predict(x)
    plt.scatter(x[:,0],x[:,1],c = y_kmeans,cmap = 'prism')
    plt.scatter(centres[:,0],centres[:,1],c="black",alpha = 1)
    plt.show()
    print("sum of squared distance")
    print(kmeans.inertia_)
    actual = df.target
    pty = purity_score(actual,pred)
    print("the value of purity score",pty)  

def ac(df):
    clustering = AgglomerativeClustering().fit(df)
    clustering = AgglomerativeClustering(affinity='euclidean',linkage='ward', n_clusters=2)
    y_pred1=clustering.fit_predict(df)

    plt.scatter(df['x'],df['y'], c=clustering.labels_, cmap='rainbow')
    plt.show()
    print(purity_score(y_true,y_pred1))
def ques5(x,df):
    #K = [ 5, 8, 10, 12, 15, 17]
    K = [2]
    for i in K:
        print("the value of k ",i)
        kmen(x,i,df)
        ac(df)

def main():
    df = load_digits()
    dataframe = pd.DataFrame(df.data)
    red_data = PCAA(dataframe)
    x = red_data.values
    ques5(x,df)
    #show_elbow_plot(x)
main()


