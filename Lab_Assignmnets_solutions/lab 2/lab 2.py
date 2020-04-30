import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
import cmath
m_data=pd.read_csv("cm.csv")
data=pd.read_csv("c.csv")
b=list(data.columns)
aa=b[2:]
#print(aa)
print(m_data.isna().sum())
a=['dispx','dispy','dispz','temperature','humidity','rain']

#m_data.interpolate(inplace=True)

for i in range(len(a)):
    m_data[a[i]].fillna(data[a[i]].median(),inplace=True)
    #m_data[a[i]].fillna(method='ffill' ,inplace=True)

for i in range(len(a)):
    print("Mean of",a[i],"is",data[a[i]].mean() )
    print("Mean of",a[i],"is",m_data[a[i]].mean() )
    print("Median of",a[i],"is",data[a[i]].median() )
    print("Median of",a[i],"is",m_data[a[i]].median() )
    print("Mode of",a[i],"is",data[a[i]].mode() )
    print("Mode of",a[i],"is",m_data[a[i]].mode() )
    print("Standard deviation of",a[i],"is",data[a[i]].std() )
    print("Standard deviation of",a[i],"is",m_data[a[i]].std() )
    print("\n")

'''for i in range(len(a)):
    plt.boxplot([m_data[a[i]],data[a[i]]], labels=("Missing Data","Original Data"))
    
    plt.show()'''
  
for i in range(len(a)):
    arry1=[]
    for j in range(945):
        q=(m_data[a[i]][j]-data[a[i]][j])*(m_data[a[i]][j]-data[a[i]][j])
        arry1.append(q)
    p=sum(arry1)/(945)
    r=(p)**0.5
    print("RMS value for attribute",a[i],"is",r) 
    print(((((data[a[i]]-m_data[a[i]])**2).sum())/len(data[a[i]]))**0.5)   
#print(((((data[a[0]]-m_data[a[0]])**2).sum())/len(data[a[0]]))**0.5)   