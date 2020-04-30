import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
import cmath
m_data=pd.read_csv("cm.csv")
data=pd.read_csv("c.csv")
#m_data.interpolate(inplace=True)
#data1=m_data['temperature'].fillna(0)
#print(data1)
#plt.hist(data1)
#plt.savefig("3.jpeg")

'''s=(m_data.groupby(['stationid'])['temperature']['dates'] )              
   
for names, groups in s:
    print("stationid =",names)
    #print(groups)
    plt.hist(groups)
    plt.xlabel("temperature")
    plt.show()'''
    
a=data['temperature'][data['stationid']=='t8']
b=data['dates'][data['stationid']=='t8']
plt.plot(b,a)
plt.show()