import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

f=pd.read_csv('atmosphere_data.csv')
x=f.iloc[:,:-1]
y=f.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

regressor=LinearRegression()
regressor.fit(x_tr_slr,y_tr_slr)

y_tr_pred=regressor.predict(x_tr_slr)
y_ts_pred=regressor.predict(x_ts_slr)

sum_err_tr=((y_tr_slr-y_tr_pred)**2).sum()
err_tr=(sum_err_tr/l_tr)**0.5

sum_err_ts=((y_ts_slr-y_ts_pred)**2).sum()
err_ts=(sum_err_ts/l_ts)**0.5


f=plt.figure(1)
plt.plot(x_tr_slr,y_tr_pred)
plt.scatter(x_tr_slr,y_tr_slr)
f.show()
g=plt.figure(2)
plt.scatter(y_ts_slr,y_ts_pred)
plt.xlabel("Actual temperature")
plt.ylabel("Predicted Temperature")
print("Predriction accuracy on training data",err_tr)
print("Predriction accuracy on test data",err_ts)

g.show()
