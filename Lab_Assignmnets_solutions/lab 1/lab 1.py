import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
import cmath
data=pd.read_csv("cm.csv")
a=['dates','stationid',	'dispx','dispy','dispz','temperature','humidity','rain']
#a=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']

print(data.isnull().sum())
print(data.isnull().sum().sum())
#print(data)
'''cnt=0
for i in range(1599):
    if (str(data[a[0]][i])=='nan'):
        cnt=cnt+1
print(cnt)'''
cnt1=0
arr=[]
for i in range(945):
    cnt1=0
    for j in range(8):
        
        if(str(data[a[j]][i])=='nan'):
          cnt1=cnt1+1          
    arr.append(cnt1) 
    
b=np.array(arr)   
for i in range(1,9):
    print('No. of tuples having',i,'missing values are',np.count_nonzero(b==i))
#plt.hist(arr)
#plt.savefig("a.jpeg")       
c=[]
d=[]
for i in range(945):
    if (str(data[a[1]][i])=='nan'):
        d.append(i)
for i in range(len(b)):
    if (b[i]>=4):
        c.append(i)
print(len(c),len(d))        
print("the number of tuples having equal to or more than 50% of attributes with missing values is",len(c))
#print(d)
data=data.drop(data.index[c])
data=data.drop(data.index[d])
print(data.isnull().sum())
print(data.isnull().sum().sum())

