import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


f=pd.read_csv('atmosphere_data.csv')
x=f.iloc[:,[0,3]]
y=f.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
train=pd.concat([x_train,y_train],axis=1)
test=pd.concat([x_test,y_test],axis=1)

x_train=train.iloc[:,:-1]
y_train=train.iloc[:,-1]

x_test=test.iloc[:,:-1]
y_test=test.iloc[:,-1]

#1-mul poly
l=2
x_tr_poly=x_train.values
x_tr_poly=x_tr_poly.reshape(len(x_tr_poly),l)
x_ts_poly=x_test.values
x_ts_poly=x_ts_poly.reshape(len(x_ts_poly),l)

y_tr_poly=y_train.values
y_tr_poly=y_tr_poly.reshape(len(y_tr_poly),1)
y_ts_poly=y_test.values
y_ts_poly=y_ts_poly.reshape(len(y_ts_poly),1)
l_tr=len(y_tr_poly)
l_ts=len(y_ts_poly)

arr=[2,3,4,5]
err_tr=[]
err_ts=[]

for p in arr:
    polynomial_features=PolynomialFeatures(degree=p)
    x_tr1_poly=polynomial_features.fit_transform(x_tr_poly)
    
    regressor=LinearRegression()
    regressor.fit(x_tr1_poly,y_tr_poly)
    
    y_tr_pred=regressor.predict(x_tr1_poly)
    y_ts_pred=regressor.predict(polynomial_features.fit_transform(x_ts_poly))
    
    sum_err_tr=((y_tr_poly-y_tr_pred)**2).sum()
    rmse_tr=(sum_err_tr/l_tr)**0.5
    err_tr.append(rmse_tr)
    
    sum_err_ts=((y_ts_poly-y_ts_pred)**2).sum()
    rmse_ts=(sum_err_ts/l_ts)**0.5
    err_ts.append(rmse_ts)

#bar graph
f1=plt.figure(1)
plt.bar(arr,err_tr,width=0.8)
plt.title('Train data')
f1.show()
f2=plt.figure(2)
plt.bar(arr,err_ts,width=0.8)
plt.title('Test data')
f2.show()

g=plt.figure(4)
plt.scatter(y_ts_poly,y_ts_pred)
#plt.ylim(1,10)
g.show()