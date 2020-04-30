import pandas as pd
import matplotlib.pyplot as plt

file=pd.read_csv('pp.csv')
arr=file.columns[:-1]

for i in arr:
    q1=file[i].quantile(0.25)
    q3=file[i].quantile(0.75)
    iqr=q3-q1
    f1=file[i]>q1-1.5*iqr
    f2=file[i]<q3+1.5*iqr
    file[i].where(f1 & f2,other=file[i].median(),inplace=True)


df_mn=file.mean()
df_std=file.std()

f3=file.copy()
f=(f3-df_mn)/df_std