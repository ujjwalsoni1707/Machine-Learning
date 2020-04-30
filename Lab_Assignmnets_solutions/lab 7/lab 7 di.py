
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from sklearn.model_selection import train_test_split
import math

df=pd.read_csv("a.csv")
x = df.iloc[:,:8].values
x=preprocessing.scale(x)
df1=pd.DataFrame(x)
df1["class"]=df["class"]


df_01=df1[df["class"]==0]
df_02=df1[df["class"]==1]
X_train0,X_test0,Y_train0,Y_test0=train_test_split(df_01.iloc[:,:8],df_01["class"],test_size=0.3,random_state=42)
X_train1,X_test1,Y_train1,Y_test1=train_test_split(df_02.iloc[:,:8],df_02["class"],test_size=0.3,random_state=42)

X_Train=pd.concat([X_train0,X_train1])
X_Test=pd.concat([X_test0,X_test1])

Y_Train=pd.concat([Y_train0,Y_train1])
Y_Test=pd.concat([Y_test0,Y_test1])


# K nearest Neighbors
kk=[1,3,5,7,9,11,13,15,17,21]
accu=[]
mat=[]
for x in kk:
    knn=KNC(n_neighbors=x)
    knn.fit(X_Train,Y_Train)
    pred=knn.predict(X_Test)
    accu.append(metrics.accuracy_score(Y_Test,pred))
    mat.append(metrics.confusion_matrix(Y_Test,pred))

mm=np.argmax(accu)
k_val=kk[mm]
#plt.plot(kk,accu)
#plt.show()
#print("Max accuracy value",accu[mm],"at k=",k_val)

#Naive Bayes

from sklearn.mixture import GaussianMixture as GMM
import math
gmm=GMM(n_components=4).fit(X_train0)
prob0=gmm.predict_proba(X_test0)
Prob0=gmm.predict_proba(X_test1)

gmm1=GMM(n_components=4).fit(X_train1)
prob1=gmm1.predict_proba(X_test0)
Prob1=gmm1.predict_proba(X_test1)





cl1=[]
cl2=[]
for i in range(len(prob1)):
    sum1=1
    sum2=1
    for x in prob1[i]:
        sum1=sum1*x
    for x in prob0[i]:
        sum2=sum2*x
    if(sum1>sum2):
        cl1.append(1)
    else:
        cl1.append(0)
    
for i in range(len(Prob1)):
    sum1=1
    sum2=1
    for x in Prob1[i]:
        sum1=sum1*x
    for x in Prob0[i]:
        sum2=sum2*x
    if(sum1>sum2):
        cl2.append(1)
    else:
        cl2.append(0)

cc=cl1+cl2
print("Accuracy Score :",metrics.accuracy_score(cc,Y_Test))
con_matx=confusion_matrix(cc,Y_Test)
print("COnfusion Matrix: \n", con_matx)