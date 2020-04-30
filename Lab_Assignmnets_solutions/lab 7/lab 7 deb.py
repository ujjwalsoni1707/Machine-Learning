import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as traintestsplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture

############################################################################################################
"""creating some functions"""
############################################################################################################

def mean(l):
     return sum(l)/len(l)

############################################################################################################

def dataRead(path):
     """returns the df"""

     return pd.read_csv(path)

############################################################################################################

def normalize_min_max(df):
     """returns the normalised attributes"""

     for i in df.columns[:-1]:
         df[i]=df[i]-df[i].min()
         df[i]=df[i]/(df[i].max()-df[i].min())

############################################################################################################

def file_save(df,new_name):
     """to save the given dataframe with new name"""

     df.to_csv(new_name)

#############################################################################################################

def standardise(df):
     """standardises the given data columns"""

     for i in df.columns[:-1]:
         df[i]=df[i]-df[i].mean()
         df[i]/=df[i].std()

##############################################################################################################

def group_this_df(df,target):
     """creates a dictionary of the dataframe passed by the class 
attributes"""

     return dict(tuple(df.groupby(target)))

################################################################################################################

def data_splitter(df_dict):
     """splits the data in 0,3 ratio and gives four dataframe outputs"""

     col=df_dict[list(df_dict.keys())[0]].columns
     #last column is the class attr.
     X=pd.DataFrame(columns=col)
     Y=pd.DataFrame(columns=col)
     for i in df_dict:
         
         a,b=traintestsplit(df_dict[i][col],test_size=0.3,random_state=20,shuffle=True)
         X = X.append(a)
         Y = Y.append(b)

     train_cols=col[:-1]
     class_col=col[-1]
     x_train = X[train_cols].astype(float)
     x_label_train = X[class_col]
     x_test = Y[train_cols].astype(float)
     x_label_test = Y[class_col]
     return x_train,x_test,x_label_train,x_label_test

############################################################################################################

def knn_classification(x_train,x_label_train,x_test,x_label_test,k):
     """this will classify and return the confusion matrix and its 
accuracy"""

     classifier= KNeighborsClassifier(n_neighbors=k)
     classifier.fit(x_train,list(x_label_train))
     x_predicted = classifier.predict(x_test)
     mat = confusion_matrix(list(x_label_test),x_predicted)
     acc = (mat[0][0]+mat[1][1])/(mat.sum())
     return mat,acc

############################################################################################################

def bayes_classification(x_train,x_label_train,x_test,x_label_test):
     """outputs the confusion matrix and accuracy of the gaussian 
model"""

     classifier = GaussianNB()
     classifier.fit(x_train,x_label_train)
     x_predict=classifier.predict(x_test)
     mat = confusion_matrix(list(x_label_test),x_predict)
     acc = (mat[0][0]+mat[1][1])/mat.sum()
     return mat,acc

############################################################################################################

def mat_acc_generator(x_train,x_label_train,x_test,x_label_test,k_values,model):
     """if there are multiple values of k to be passed this function is 
used
         it will simply return multiple confusion_matrices and accuracies 
corresponding to each"""

     matrices=[]
     accuracies=[]
     for k in k_values:
         a,b=model(x_train,x_label_train,x_test,x_label_test,k)
         matrices.append(a)
         accuracies.append(b)

     return matrices,accuracies

############################################################################################################

def sample_mean(df):
     return df.mean().values

#############################################################################################################

def sample_cov_mat(df):
     return np.array(df.cov())

#############################################################################################################

def eigen_values(cov_mat):
     return np.linalg.eig(cov_mat)

#############################################################################################################

def prob(x, w, mean, cov):
     p = 0
     for i in range(len(w)):
         p += w[i] * scipy.stats.multivariate_normal.pdf(x, mean[i], 
cov[i], allow_singular=True)

     return p

#############################################################################################################

def gmmbayes(df,k):
     df_0 = group_this_df(df,'class')[0]
     df_1= group_this_df(df,'class')[1]

     X_train0, X_test0, y_train0, y_test0 = traintestsplit(df_0[df_0.columns[:-1]],df_0['class'],test_size = 0.3, random_state=42)
     X_train1, X_test1, y_train1, y_test1 = traintestsplit(df_1[df_0.columns[:-1]],df_1['class'],test_size = 0.3, random_state=42)

     test = np.concatenate((X_test0, X_test1))
     pred = np.concatenate((y_test0, y_test1))

     gmm = GaussianMixture(n_components=k)
     gmm.fit(X_train0)

#After our model has converged, the weights, means, and covariances should be solved! We can print them out.

#    print("gmm mean_ ", gmm.means_)
     gmm2 = GaussianMixture(n_components=k)
     gmm2.fit(X_train1)

     print('for class 0')
     
     print('means\n',gmm.means_);print('covariances\n',gmm.covariances_);print('weights\n',gmm.weights_)
     print('for class 1')
     
     print('means\n',gmm2.means_);print('covariances\n',gmm2.covariances_);print('weights\n',gmm2.weights_)

     ypred = []
     for i in test:
         ypred.append(  0 if prob(i, gmm.weights_, gmm.means_, gmm.covariances_)\
                            > prob(i, gmm2.weights_, gmm2.means_, 
gmm2.covariances_) else 1 )
     print("Accuracy for GMM  Bayes Classifier: ")
     print(accuracy_score(pred, ypred))
     print(confusion_matrix(pred, ypred))
     print()

###############################################################################################################

path="a.csv"
df0 = dataRead(path)
standardise(df0)

df_dict = group_this_df(df0,'class')

x_train,x_test,x_label_train,x_label_test = data_splitter(df_dict)

k_values = [1,3,5,7,9,11,13,15,17,21]

matrices,accuracies=mat_acc_generator(x_train,x_label_train,x_test,x_label_test,k_values,knn_classification)

q=[2,4,8,16]

gmmbayes(df0,4)


-- 
Dipanshu Verma
CSE (batch 2k18)
B18054