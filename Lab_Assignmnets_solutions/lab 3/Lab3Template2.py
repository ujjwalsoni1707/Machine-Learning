from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

path_to_file="pp.csv"
df=pd.read_csv(path_to_file)
df=df.drop['quality']    
a=['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality']
print(df.head(3))
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

