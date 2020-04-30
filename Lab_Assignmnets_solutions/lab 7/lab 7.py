# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:16:31 2019

@author: Ujjwal Soni
"""
from sklearn import mixture
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("a.csv")
data1=pd.read_csv("a.csv")
######################################################
#a=data["Class"]
#data=data.drop(columns="Class")   
#    #print(data.head(10))
#scaler = StandardScaler() 
#scaled_values = scaler.fit_transform(data) 
#data.loc[:,:] = scaled_values 
#data["Class"]=a
#a=data1["Class"]
#data1=data1.drop(columns="Class")   
##print(data.head(10))
#scaler = StandardScaler() 
#scaled_values = scaler.fit_transform(data1) 
#data1.loc[:,:] = scaled_values 
#data1["Class"]=a    
##################################################    
y=data["class"]
x=data.drop('class',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
acu_scr=[]
print("Before PCA")
classifier=  mixture.GaussianMixture(n_components=6)
classifier.fit(x_train, y_train)
x_pred = classifier.predict(x_test)
con_matx=confusion_matrix(y_test, x_pred)
acc=((con_matx[0][0]+con_matx[1][1])/con_matx.sum())
    #print("confusion matrix =")
    #print(con_matx)
#acu_scr.append((con_matx[0][0]+con_matx[1][1])/sum(sum(con_matx)))
#print(acu_scr) 
print("The Confusion Matrix Is: \n",con_matx)
print("The Accuracy Score is: ",acc)
print("\n")
print("After PCA")
#################################################################################################
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data)
df1 = pd.DataFrame(data = principalComponents)
df = pd.concat([df1, data[['class']]], axis = 1)
#print(df)
#################################################################################################
y=df["class"]
x=df.drop('class',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
acu_scr=[]
classifier= mixture.GaussianMixture(n_components=6)
classifier.fit(x_train, y_train)
x_pred = classifier.predict(x_test)
con_matx=confusion_matrix(y_test, x_pred)
acc=((con_matx[0][0]+con_matx[1][1])/con_matx.sum())
    #print("confusion matrix =")
    #print(con_matx)
#acu_scr.append((con_matx[0][0]+con_matx[1][1])/sum(sum(con_matx)))
#print(acu_scr) 
print("The Confusion Matrix Is: \n",con_matx)
print("The Accuracy Score is: ",acc)
##############################################################################3#################

