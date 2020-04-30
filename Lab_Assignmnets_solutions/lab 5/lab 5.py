import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
data=pd.read_csv("a.csv")

y=data["class"]
x=data.drop('class',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
acu_scr=[]
a=[1, 3, 5, 7, 9, 11, 13, 15, 17, 21]
for i in range(len(a)):
    classifier = KNeighborsClassifier(n_neighbors=a[i])
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    con_matx=confusion_matrix(y_test, y_pred)
    #print("confusion matrix =")
    #print(con_matx)
    acu_scr.append((con_matx[0][0]+con_matx[1][1])/sum(sum(con_matx)))
print(acu_scr) 



