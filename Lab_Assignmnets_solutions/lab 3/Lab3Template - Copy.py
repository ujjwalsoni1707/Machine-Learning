import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(path_to_file):
	return(pd.read_csv(path_to_file))
    
def show_box_plot(attribute_name,dataframe):
    plt.boxplot(dataframe[attribute_name])
    plt.show()
    print(attribute_name,"with outliers")
    pass

def replace_outliers(a,df,i):
    x=df.quantile(0.25)
    #print(x)
    y=df.quantile(0.75)
    b=x[i]
    c=y[i]
    k=0    
    cnt=0
    while(k<945):
        if (((b-1.5*(c-b)) > df[a][k]) or ((c+1.5*(c-b)) < df[a][k])):
            cnt=cnt+1
            df.replace(to_replace =df[a][k],value =df[a].median(),inplace=True) 
        k=k+1    
    print(cnt)
    plt.boxplot(df[a])
    plt.show() 
    print(a,"without outliers")           
    pass

def ran_ge(df,a):
    #print("\n")
   # print("min of",a,"is",df[a].min())
    #print("max of",a,"is",df[a].max())
    pass

def min_max_normalization(df):
    df1=((df-df.min())/(df.max()-df.min()))*20
    pass

def standardize(df):
    df2=(df-df.mean())/df.std()
    #print("\n Values of mean and std after Standardiztion")
    #print(df2.mean())
    #print(df2.std())
    #print("\n")
    pass

def main():
    path_to_file="qq.csv"
    df=read_data(path_to_file)
    #print(df)
    df=df.drop(columns="dates")
    df=df.drop(columns="stationid")
    a=list(df.columns)
    i=0
    '''while(i<(len(a))):
        show_box_plot(a[i],df)
        replace_outliers(a[i],df,i)
        ran_ge(df,a[i])
        i=i+1'''
    #min_max_normalization(df)    
    #standardize(df)
   # print("Values of mean and std before Standardiztion")
    #print(df.mean())
    #print(df.std())
    df4=df.copy()
    for i in df4.columns:
        col=df4[i]
        q1=col.quantile(0.25)
        q3=col.quantile(0.75)
        iqr=q3-q1
        print(df4.loc[(col<(q1-1.5*iqr))|(col>(q3+1.5*iqr)),i].count())
        
    return 0    
if __name__=="__main__":
	main()