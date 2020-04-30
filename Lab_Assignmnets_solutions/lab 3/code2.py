import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('pp.csv')
arr=data.columns[0:-1]

for i in arr:
    q1=data[i].quantile(0.25)
    q3=data[i].quantile(0.75)
    iqr=q3-q1
    f1=data[i]>q1-1.5*iqr
    f2=data[i]<q3+1.5*iqr
    data[i].where(f1 & f2,other=data[i].median(),inplace=True)

#data.boxplot(column=list(arr))
    
arr_min=[]
for i in arr:
    arr_min.append(data[i].min())

arr_max=[]
for i in arr:
    arr_max.append(data[i].max())


f1=data.copy()
for i in range(len(arr)):
    f1[arr[i]]=(f1[arr[i]]-arr_min[i])/(arr_max[i]-arr_min[i]);

f2=data.copy()
for i in range(len(arr)):
    f2[arr[i]]=((f2[arr[i]]-arr_min[i])/(arr_max[i]-arr_min[i]))*(20);