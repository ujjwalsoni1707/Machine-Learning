import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


f=pd.read_csv('atmosphere_data.csv')
a=['humidity','pressure','rain','lightAvg','lightMax','moisture','temperature']
for i in range(7):
    print("corr of ",a[6],"with",a[i],"is",f[a[6]].corr(f[a[i]]))
x=f.iloc[:,[0,3]]
y=f.iloc[:,-1]

x_tr_slr,x_ts_slr,y_tr_slr,y_ts_slr=train_test_split(x,y,test_size=0.3,random_state=42)

#1-mul linear
l=2


regressor=LinearRegression()
regressor.fit(x_tr_slr,y_tr_slr)

y_tr_pred=regressor.predict(x_tr_slr)
y_ts_pred=regressor.predict(x_ts_slr)

sum_err_tr=((y_tr_slr-y_tr_pred)**2).sum()
err_tr=(sum_err_tr/l_tr)**0.5

sum_err_ts=((y_ts_slr-y_ts_pred)**2).sum()
err_ts=(sum_err_ts/l_ts)**0.5

g=plt.figure(1)
plt.scatter(y_ts_slr,y_ts_pred)
plt.xlabel("ACtual temperature")
plt.ylabel("Predicted temperature")
g.show()


print("Predriction accuracy on training data",err_tr)
print("Predriction accuracy on test data",err_ts)

fig = plt.figure()
ax = fig.add_subplot(111 , projection = '3d')
ax.scatter(x_tr_slr['humidity'],x_tr_slr['lightAvg'],y_tr_slr,color="g",marker=".")
#fig1 = plt.gcf()
#fig1.set_size_inches(10, 10)
ax.plot_trisurf(x_tr_slr['humidity'],x_tr_slr['lightAvg'],y_tr_pred,alpha=0.5)
ax.scatter(x_tr_slr['humidity'],x_tr_slr['lightAvg'],y_tr_pred,color="r",marker="*")
plt.show()
