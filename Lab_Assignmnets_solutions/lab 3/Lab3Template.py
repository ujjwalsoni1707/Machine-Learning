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
    y=df.quantile(0.75)
    b=x[i]
    c=y[i]
    k=0    
    while(k<1599):
        if (((b-1.5*(c-b)) > df[a][k]) or ((c+1.5*(c-b)) < df[a][k])):
            df.replace(to_replace =df[a][k],value =df[a].median(),inplace=True) 
        k=k+1    
    plt.boxplot(df[a])
    plt.show() 
    print(a,"without outliers")           
    pass

def ran_ge(df,a):
    print("\n")
    print("min of",a,"is",df[a].min())
    print("max of",a,"is",df[a].max())
    pass

def min_max_normalization(df):
    df1=(df-df.min())/(df.max()-df.min())
    
    pass

def standardize(df):
    df2=(df-df.mean())/df.std()
    a=['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality']
    print(df2.mean())
    print(df2.std())
    print("\n")
    pass

def main():
    path_to_file="pp.csv"
    df=read_data(path_to_file)
    
    a=['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality']
    i=0
    while(i<11):
        show_box_plot(a[i],df)
        replace_outliers(a[i],df,i)
        ran_ge(df,a[i])
        i=i+1
    min_max_normalization(df)    
    standardize(df)
    i=0
    while(i<11):
        print("Mean and Std of",a[i],"are",df[a[i]].mean(),"and",df[a[i]].std())
        i=i+1
        
    return 0    
if __name__=="__main__":
	main()