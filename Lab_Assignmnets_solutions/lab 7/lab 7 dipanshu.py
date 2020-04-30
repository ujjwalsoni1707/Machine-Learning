import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import mixture

# Import or Load the data
def load_dataset(path_to_file):
    data = pd.DataFrame(pd.read_csv(path_to_file))
    return data

def outliers_detection(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3-q1
    lower, upper = q1-(iqr*1.5), q3+(iqr*1.5)
    
    outliers = [x for x in data if x<lower or x>upper]
    return outliers

#def missing_values(function_parameters):

    
def encoding(data):
    data = data.apply(preprocessing.LabelEncoder().fit_transform)
    return data
    
def normalization(data):
    scaler = MinMaxScaler()
    scaler = scaler.fit(data)
    return scaler.transform(data)

def standardization(data):
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    return scaler.transform(data)
    
def dimensionality_reduction(data,n):
    pca = PCA(n_components=n)
    dr = pca.fit(data)  
    data_dr = dr.fit_transform(data)
    return data_dr

def shuffle_(X):
    s_data = shuffle(X,random_state=42)
    return s_data

def train_test_split_(data,y):
    print(data.shape)
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.30, random_state=42)
    return X_train, X_test, y_train, y_test

# Perform classification
def classification(xtrain,ytrain,xtest,n):
    neigh = KNeighborsClassifier(n_neighbors=n)
    neigh.fit(xtrain, ytrain)
    
    predict = neigh.predict(xtest)
    return predict

def percentage_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)*100

def confusion_matrix_(y_true,y_pred):
    return confusion_matrix(y_true, y_pred)

###################################################################################

data = load_dataset("a.csv")
print(data.head())

train = data.loc[:,'pregs':'Age']
test = data['class']

train = pd.DataFrame(standardization(train))

X_train, X_test, y_train, y_test = train_test_split(train,test)

k = [1,3,5,7,9,11,13,15,17,19,21]
acc = []
for i in k:
    y_pred = classification(X_train,y_train,X_test,i)
    print("For k = ",i)
    #print("Confusion Matrix ", confusion_matrix_(y_test,y_pred))
    print("Percentage Accuracy ", percentage_accuracy(y_test, y_pred),"%")
    acc.append(percentage_accuracy(y_test, y_pred))

plt.plot(k,acc)
plt.xticks(k)
plt.show()

print("GMM Classifier")
acc = []
q = [1,2,4,8,16]
for i in q:
    gmm = mixture.GaussianMixture(n_components=i)
    y_pred = gmm.fit(X_train, y_train).predict(X_test)
    #print(y_pred)
    #print("Confusion Matrix ", confusion_matrix_(y_test,y_pred))
    print("Percentage Accuracy ", percentage_accuracy(y_test, y_pred),"%")
    acc.append(percentage_accuracy(y_test, y_pred))
plt.plot(q,acc)
plt.show()

l2 = [1,2,3,4,5,6,7,8]


for i in q:
    acc2 = []
    for l in range(1,9):
        #print("For l = ",l)
        trainx = dimensionality_reduction(train,l)
        X_train, X_test, y_train, y_test = train_test_split(trainx,test)
    
        acc1 = []
    
        """#part1
        for i in k:
            y_pred1 = classification(X_train,y_train,X_test,i)
            print("For k = ",i)
            #print("Confusion Matrix ", confusion_matrix_(y_test,y_pred1))
            #print("Percentage Accuracy ", percentage_accuracy(y_test, y_pred1),"%")
            acc1.append(percentage_accuracy(y_test, y_pred1))
        plt.ylim(50,80)
        plt.plot(k,acc1)
        plt.show()"""
    
        #part2
        gmm1 = mixture.GaussianMixture(n_components=i)
        y_pred2 = gmm1.fit(X_train,y_train).predict(X_test)
        #print("Bayes Classifier")
        #print("Confusion Matrix ", confusion_matrix_(y_test,y_pred2))
        #print("Percentage Accuracy ", percentage_accuracy(y_test, y_pred2),"%")
        acc2.append(percentage_accuracy(y_test, y_pred2))
        
        #print()

    #plt.ylim(50,80)
    plt.ylabel("Gmm accuracy")
    plt.xlabel("Dimension of data")
    plt.plot(l2,acc2)
    plt.show()

'''import scipy.stats

train = data.loc[:,'pregs':'Age']
test = data['class']

train = pd.DataFrame(standardization(train))

X_train, X_test, y_train, y_test = train_test_split(train,test)

df = data.groupby('class') 

#class0
data0 = df.get_group(0)
#class1
data1 = df.get_group(1)

d0 = dimensionality_reduction(data0,2)
d1 = dimensionality_reduction(data1,2)

d0 = np.reshape(d0,(2,500))
d1 = np.reshape(d1,(2,268))

plt.scatter(d0[0],d0[1],marker='x')
plt.scatter(d1[0],d1[1],marker='x')
plt.show()'''