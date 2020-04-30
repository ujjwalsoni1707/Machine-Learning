

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import acf

df=pd.read_csv('daily-min-temperatures.csv')


l=np.arange(0,len(df['Date']))
plt.plot(l,df['Temp'])
plt.show()

li=list(df['Temp'])
l_new=li[0:len(li)-1]
correl=pearsonr(l_new,li[1:len(li)])
print(correl)

c=acf(li,nlags=30)
ll=np.arange(1,32)

plt.plot(ll,c)
plt.show()


from sklearn.metrics import mean_squared_error as mse

l_train=li[0:len(li)-7]
l_test=li[len(li)-7:len(li)]

correl=acf(l_train,nlags=31)
lag=np.argmax(correl)+1

l_pred=li[len(li)-7-lag:len(li)-lag]

print('The mean squared error is',mse(l_test,l_pred))



from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

model = AR(l_train)
model_fit = model.fit()
print('Lag: ', model_fit.k_ar)
print('Coefficients: ', model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(l_train), end=len(l_train)+len(l_test)-1, dynamic=False)
error = mean_squared_error(l_test, predictions)
print('Test MSE: ' , error)