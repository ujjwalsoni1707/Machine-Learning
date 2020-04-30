import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def load_dataset(file_name):
    data=pd.read_csv(file_name)
    return data
def normalization(df):
    scaler = MinMaxScaler() 
    scaled_values = scaler.fit_transform(df) 
    df.loc[:,:] = scaled_values 
    return df
def standardization(df):
    a=df["Class"]
    df=df.drop(columns="Class")   
    #print(data.head(10))
    scaler = StandardScaler() 
    scaled_values = scaler.fit_transform(df) 
    df.loc[:,:] = scaled_values 
    df["Class"]=a
    
    return df

def _train_test_split_(df):
    y=df["Class"]
    x=df.drop('Class',axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
    arr=[]
    arr.append(x_train)
    arr.append(x_test)
    arr.append(y_train)
    arr.append(y_test)
    return arr
def classification_and_confusion_matrix(arr):
    acu_scr=[]
    a=[1, 3, 5, 7, 9, 11, 13, 15, 17, 21]
    for i in range(len(a)):
        classifier = KNeighborsClassifier(n_neighbors=a[i])
        classifier.fit(arr[0], arr[2])
        y_pred = classifier.predict(arr[1])
        con_matx=confusion_matrix(arr[3], y_pred)
        #print("confusion matrix =")
        #print(con_matx)
        acu_scr.append((con_matx[0][0]+con_matx[1][1])/sum(sum(con_matx)))
    return acu_scr 


data1=load_dataset("a.csv")
data2=load_dataset("a.csv")
data3=load_dataset("a.csv")

data2=normalization(data2)

data3=standardization(data3)

arr1=_train_test_split_(data1)
arr2=_train_test_split_(data2)
arr3=_train_test_split_(data3)

acu_scr1=classification_and_confusion_matrix(arr1)
acu_scr2=classification_and_confusion_matrix(arr2)
acu_scr3=classification_and_confusion_matrix(arr3)
print(acu_scr1,acu_scr2,acu_scr3)
a=[1, 3, 5, 7, 9, 11, 13, 15, 17, 21]




plt.bar(a,acu_scr2,label="Normalised Data",align="edge",width=0.8)
plt.bar(a,acu_scr3,label="Standardized Data",align="edge",width=-0.8)
plt.bar(a,acu_scr1,label="Original Data",align="center",width=0.5)
plt.xticks(a)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
plt.savefig('destination_path1.eps', format='eps')
plt.plot()
plt.show()
plt.plot(a,acu_scr1,color="red",label="Original Data")
plt.plot(a,acu_scr2,color="yellow",label="Normalised Data")
plt.plot(a,acu_scr3,label="Standardised Data")
plt.xticks(a)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
plt.savefig('destination_path2.eps', format='eps')



    
    
    
    
    
    
    
    
    
    