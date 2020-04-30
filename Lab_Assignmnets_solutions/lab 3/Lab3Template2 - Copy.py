from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

path_to_file="qq.csv"
df=read_data(path_to_file)
df=df.drop(columns="dates")
df=df.drop(columns="stationid")
#print(df)
df1=df
df2=df

scaler = MinMaxScaler() 
scaled_values = scaler.fit_transform(df1) 
df1.loc[:,:] = scaled_values 

print(df1.head(3))   

scaler = StandardScaler() 
scaled_values = scaler.fit_transform(df2) 
df2.loc[:,:] = scaled_values 
print(df2.head(3))
print(df2.mean() )  
print(df2.std())