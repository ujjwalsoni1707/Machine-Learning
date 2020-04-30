import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("a.csv")
data1=pd.read_csv("a.csv")
#################################################################################################
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data)
df1 = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
df = pd.concat([df1, data[['Class']]], axis = 1)
#print(df)
#################################################################################################
print("After Apllying PCA")
y=df["Class"]
x=df.drop('Class',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
acu_scr=[]
a=[1, 3, 5, 7, 9, 11, 13, 15, 17, 21]
for i in range(len(a)):
    classifier = KNeighborsClassifier(n_neighbors=a[i])
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    con_matx=confusion_matrix(y_test, y_pred)
    
    print("confusion matrix for k =",a[i],"is\n",con_matx)
    #print(con_matx)
    acu_scr.append((con_matx[0][0]+con_matx[1][1])/sum(sum(con_matx)))
for i in range(len(acu_scr)):
    print("Accuracy score for k=",a[i],"is",acu_scr[i])
####################################################################################################
print("\n\n")
print("BEfore Applying PCA")
y=data["Class"]
x=data.drop('Class',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
acu_scr=[]
a=[1, 3, 5, 7, 9, 11, 13, 15, 17, 21]
for i in range(len(a)):
    classifier = KNeighborsClassifier(n_neighbors=a[i])
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    con_matx=confusion_matrix(y_test, y_pred)
    print("confusion matrix for k =",a[i],"is\n",con_matx)
    #print(con_matx)
    acu_scr.append((con_matx[0][0]+con_matx[1][1])/sum(sum(con_matx)))
for i in range(len(acu_scr)):
    print("Accuracy score for k=",a[i],"is",acu_scr[i])