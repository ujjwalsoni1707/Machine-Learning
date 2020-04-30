import pandas as pd
import matplotlib.pyplot as plt

file=pd.read_csv('winequality_red_original.csv')

#1
arr=file.columns[:-1]
#file.boxplot(column=list(arr))
#plt.show()
#
#arr_out=[]
#for i in arr:
#    q1=file[i].quantile(0.25)
#    q3=file[i].quantile(0.75)
#    iqr=q3-q1
#    f1=file[i]<q1-1.5*iqr
#    f2=file[i]>q3+1.5*iqr
#    arr_out.append(file[i].where(f1 | f2))
#
#for i in range(len(arr)):
#    print('\n',arr[i],'\n')
#    print(arr_out[i].value_counts())
#    print(arr_out[i].value_counts().sum())

q1=file.quantile(0.25)
q3=file.quantile(0.75)
iqr=q3-q1
f1=file>q1-1.5*iqr
f2=file<q3+1.5*iqr
file.where(f1 & f2,other=file.median(),axis=1,inplace=True)

file.boxplot(column=list(arr))

