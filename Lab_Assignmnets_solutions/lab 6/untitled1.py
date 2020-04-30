# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:45:01 2019

@author: Ujjwal Soni
"""

a=df["Class"]
df=df.drop(columns="Class")   
    #print(data.head(10))
scaler = StandardScaler() 
scaled_values = scaler.fit_transform(df) 
df.loc[:,:] = scaled_values 
df["Class"]=a