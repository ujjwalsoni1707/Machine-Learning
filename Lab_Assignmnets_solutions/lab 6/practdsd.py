# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:13:52 2019

@author: Ujjwal Soni
"""

from sklearn.cluster import KMeans

sklearn_pca = PCA(n_components = 2)

Y_sklearn = sklearn_pca.fit_transform(tf_idf_array)

kmeans = KMeans(n_clusters=3, max_iter=600, algorithm = 'auto')

fitted = kmeans.fit(Y_sklearn)

prediction = kmeans.predict(Y_sklearn)
plt.figure(figsize = (10,8))

from scipy.spatial.distance import cdist

def plot_kmeans(kmeans, X, n_clusters=3, rseed=0, ax=None):

    labels = kmeans.fit_predict(X)



    # plot the input data

    ax = ax or plt.gca()

    ax.axis('equal')

    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)



    # plot the representation of the KMeans model

    centers = kmeans.cluster_centers_

    radii = [cdist(X[labels == i], [center]).max()

             for i, center in enumerate(centers)]

    for c, r in zip(centers, radii):

        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))

        

plot_kmeans(kmeans, Y_sklearn)
