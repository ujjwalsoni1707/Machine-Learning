import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg as la
import sys
from sklearn.linear_model import LinearRegression

arr=np.random.multivariate_normal(mean=[0,0],cov=[[7,10],[10,18]],size=1000)
#print("size of array=" ,sys.getsizeof(arr[0]))
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
mse = (se.sum()/len(se))**(1/2)

#arr = np.random.multivariate_normal(mean=[0,0] ,cov=[[7,10],[10,18]],size=1000)
x = arr[:,0]
y = arr[:,1];
corrMatrix = np.cov(arr.T)
a,b = la.eig(corrMatrix);#COVR_MATRIX
print("Eigen Values",a)
eigen1 = b[:,0]
eigen2 = b[:,1]
#print(eigen1)
#print(eigen2)
#print(a)

constructed = np.matmul(arr,b)

mod_cov=np.cov(constructed.T)#cov

temparray = np.linalg.inv(b)
reconstructed = np.matmul(constructed,temparray)


#constructed = np.matmul(b.T,arr.T)
#dataReconstructed=np.matmul(data_mod,b)
plt.scatter(x,y)
plt.quiver(0,0,b[0][0],b[0][1],scale=50)
plt.quiver(0,0,b[1][0],b[1][1],scale=50)
plt.plot()



plt.scatter(data_mod[:,0],data_mod[:,1],color="red")
t1 = np.array([[-1*cs],[sn]])
t2 = np.array([[-1*cs,sn]])
datalessImportant = np.matmul(arr,t1)
dataAlongper = np.matmul(datalessImportant,t2)

plt.scatter(dataAlongper[:,0],dataAlongper[:,1],color="green")
plt.show()

err1= (dataAlongper-arr)**2
err2 = (data_mod-arr)**2
print((err1.sum())/1000)
print((err2.sum())/1000)
#plt.quiver(*org,*eg1,scale=1)
#plt.quiver(*org,*eg2,scale=20)