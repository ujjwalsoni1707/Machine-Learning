import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
import scipy

def load_dataset(path_to_file):
    df=pd.read_csv(path_to_file)
    return df

def prob(x, w, mean, cov):
     p = 0
     for i in range(len(w)):
         p += w[i] * scipy.stats.multivariate_normal.pdf(x, mean[i], cov[i], allow_singular=True)
         return p
def dimensionality_reduction(df,a):
     X=df.drop(columns='class')
     pca=PCA(n_components=a)
     df2=pca.fit_transform(X)
     return df2
def train_test_data(df):
     X=df.drop(columns='class')
     Y=df['class']
     X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.3, random_state=42)
     return X_test,X_train,Y_test,Y_train

def classification(X_train, Y_train, n_neighbours):
    clf = KNeighborsClassifier(n_neighbours)
    clf.fit(X_train, Y_train)
    return clf

def bayes_classifier(X_train,Y_train):
     gnb=GaussianNB()
     gnb.fit(X_train,Y_train)
     return gnb

def percentage_accuracy(Y_pred, Y_test):
     classification_accuracy = sklearn.metrics.accuracy_score(Y_pred, Y_test)
     return 100*classification_accuracy


def confusion_matrix(Y_pred, Y_test):
     con_mat = sklearn.metrics.confusion_matrix(Y_pred, Y_test)
     return con_mat

def bayes_classifier_2(k,df):
     df_0 = df[df['class']==0]
     df_1 = df[df['class']==1]
     X_train0, X_test0, y_train0, y_test0 = train_test_data(df_0)
     X_train1, X_test1, y_train1, y_test1 = train_test_data(df_1)
     test = np.concatenate((X_test0, X_test1))
     pred = np.concatenate((y_test0, y_test1))
     gmm = GaussianMixture(n_components=k)
     gmm.fit(X_train0)


     gmm2 = GaussianMixture(n_components=k)
     gmm2.fit(X_train1)
     print('for class 0')
     print('means\n',gmm.means_);print('covariances\n',gmm.covariances_);print('weights\n',gmm.weights_)
     print('for class 1')
     print('means\n',gmm2.means_);print('covariances\n',gmm2.covariances_);print('weights\n',gmm2.weights_)
     ypred = []
     for i in test:
         ypred.append(  0 if prob(i, gmm.weights_, gmm.means_, gmm.covariances_)\
                      > prob(i, gmm2.weights_, gmm2.means_, gmm2.covariances_) else 1 )
     print("Accuracy for GMM  Bayes Classifier: ")
     print(percentage_accuracy(pred, ypred))
     print(confusion_matrix(pred, ypred))
     print()


df=load_dataset("a.csv")
a=df.columns
#print(df)

df2=df
gg=df
count1 = df[df['Age']>=0]['Age'].count()
head= int(count1*7/10)
dfX_test,dfX_train,dfY_test,dfY_train=train_test_data(df)



dfh= pd.concat([dfX_train,dfY_train], axis=1)
dft= pd.concat([dfX_test,dfY_test],axis=1)


bayes_classifier_2(4,df)

print(" GMM bayes classification")
bay = bayes_classifier(dfX_train,dfY_train)
dfbY_pred = bay.predict(dfX_test)
con_mat = confusion_matrix(dfbY_pred, dfY_test)
print(con_mat)
print(percentage_accuracy(dfbY_pred, dfY_test))





























