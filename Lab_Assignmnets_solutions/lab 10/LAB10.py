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

def GMM(x,i):
    gmm = gaussian_mixture.GaussianMixture(n_components = i).fit(x)
    labels = gmm.predict(x)
    plt.scatter(x[:,0],x[:,1],c = labels,cmap = 'viridis')
    plt.show()

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

def ques5(x,df):
    #K = [ 5, 8, 10, 12, 15, 17]
    K = [10]
    for i in K:
        print("the value of k ",i)
        kmen(x,i,df)
        print("the value of no of component in GMM ",i)
        GMM(x,i)

def show_elbow_plot(x):
    model = KMeans()
    visualizer = KElbowVisualizer(model,k=(4,12))
    visualizer.fit(x)
    visualizer.show()
def main():
    df = load_digits()
    dataframe = pd.DataFrame(df.data)
    red_data = PCAA(dataframe)
    x = red_data.values
    ques5(x,df)
    #show_elbow_plot(x)
main()


