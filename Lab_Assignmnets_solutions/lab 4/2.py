# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 19:22:49 2019

@author: Ujjwal Soni
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg as la
import sys
from sklearn.linear_model import LinearRegression
arr=np.random.multivariate_normal(mean=[4,8],cov=[[1,4],[4,10]],size=1000)
x=arr[:,0]
y=arr[:,1]

plt.scatter(x,y)
plt.show()
regression_model = LinearRegression()
x=x.reshape((len(x),1))
y=y.reshape((len(y),1))
regression_model.fit(x, y)
slope=regression_model.coef_
theta=math.atan(slope)
cs=math.cos(theta)
sn=math.sin(theta)

arr1=np.array([[cs],[sn]])
data1d = np.matmul(arr,arr1)
array2 = np.array([[cs,sn]])
data_mod = np.matmul(data1d,array2)
se = ((arr-data_mod)**2)
mSquareError = (se.sum()/len(se))**(1/2)


x = arr[:,0]
y = arr[:,1];
corrMatrix = np.corrcoef(arr.T)
a,b = la.eig(corrMatrix);
eigen1 = b[:,0]
eigen2 = b[:,1]
constructed = np.matmul(arr,b)
temparray = np.linalg.inv(b)
reconstructed = np.matmul(constructed,temparray)


constructed = np.matmul(b.T,arr.T)
dataReconstructed=np.matmul(data_mod,b)
plt.scatter(x,y)
plt.quiver(0,0,b[0][0],b[0][1])
plt.quiver(0,0,b[1][0],b[1][1])
plt.plot()
plt.plot(data_mod[:,0],data_mod[:,1])






























