import pandas as pd
import numpy as np
import sklearn.linear_model as li
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import scipy.stats as st
from mpl_toolkits.mplot3d import Axes3D
import math
import operator

def readfile(path):
    df = pd.read_csv(path)
    return df
def trn_tst(df):
    X_train, X_test = model_selection.train_test_split(df, test_size=0.3, random_state=42,shuffle=True)
    return X_train,X_test

def linear_reg(X_train,X_test):
    X = X_train[["pressure"]]
    Y = X_train["temperature"]
    test = X_test[["pressure"]]
    actual = X_test["temperature"]
    reg = li.LinearRegression()
    reg.fit(X,Y)
    pred_qual = reg.predict(test)
    pred_trn_err = math.sqrt(metrics.mean_squared_error(Y,reg.predict(X)))
    pred_tst_err = math.sqrt(metrics.mean_squared_error(actual,reg.predict(test)))
    print("RMSE of training data is ",pred_trn_err)
    print("RMSE of test data is ",pred_tst_err)
    plt.scatter(X ,Y , marker = "*",c = "r")
    plt.plot(X, reg.predict(X))
    plt.xlabel("pressure")
    plt.ylabel("temperature")
    plt.show()
    plt.scatter(actual,pred_qual)
    plt.xlabel("test quality")
    plt.ylabel("pred quality")
    plt.show()
    
def poly_reg(X_train,X_test):
    X = X_train["pressure"].values
    X = X.reshape(-1,1)
    Y = X_train["temperature"].values
    test = X_test[["pressure"]]
    actual = X_test["temperature"]
    p=[2,3,4,5]
    RMSE_trn=[]
    RMSE_tst=[]
    for i in p:
        print("degree of polynomial is ",i)
        poly = PolynomialFeatures(degree = i )
        x_poly = poly.fit_transform(X)
        test_poly = poly.fit_transform(test)
        poly.fit(x_poly,Y)
        linreg = li.LinearRegression()
        linreg.fit(x_poly,Y)
        pred_qual = linreg.predict(test_poly)
        pred_trn_err = math.sqrt(metrics.mean_squared_error(Y,linreg.predict(x_poly)))
        pred_tst_err = math.sqrt(metrics.mean_squared_error(actual,linreg.predict(test_poly)))
        print("RMSE of training data is ",pred_trn_err)
        print("RMSE of test data is ",pred_tst_err)
        RMSE_trn.append(pred_trn_err)
        RMSE_tst.append(pred_tst_err)
        sort_axis = operator.itemgetter(0)
        sorted_zip = sorted(zip(X,linreg.predict(x_poly)), key=sort_axis)
        x, y_poly_pred = zip(*sorted_zip)

        plt.scatter(X ,Y , marker = "*",c = "r")
        plt.plot(x, y_poly_pred)
        plt.xlabel("pressure")
        plt.ylabel("temperature")
        plt.show()
        plt.scatter(actual,pred_qual)
        plt.xlabel("test temperature")
        plt.ylabel("pred quality")
        plt.show()
    plt.bar(p,RMSE_trn,color="orange")
    plt.xlabel("degree of polynomial")
    plt.ylabel("RMSE of train data")
    plt.title("bar graph for train data error")
    plt.show()
    plt.bar(p,RMSE_tst,color="green")
    plt.xlabel("degree of polynomial")
    plt.ylabel("RMSE of test data")
    plt.title("bar graph for test data error")
    plt.show() 
        
def mul_lin_reg(X_train,X_test):
    pd.set_option('mode.chained_assignment', None)
    Y = X_train["temperature"]
    #print(Y)
    X_train.drop(columns=["temperature"],inplace = True)
    #print(X_train)
    actual = X_test["temperature"]
    X_test.drop(columns=["temperature"],inplace = True)
    reg = li.LinearRegression()
    reg.fit(X_train,Y)
    pred_qual = reg.predict(X_test)
    pred_trn_err = math.sqrt(metrics.mean_squared_error(Y,reg.predict(X_train)))
    pred_tst_err = math.sqrt(metrics.mean_squared_error(actual,reg.predict(X_test)))
    print("RMSE of training data is ",pred_trn_err)
    print("RMSE of test data is ",pred_tst_err)
    plt.scatter(actual,pred_qual)
    plt.xlabel("test quality")
    plt.ylabel("pred quality")
    plt.show()

def mul_poly_reg(X_train,X_test):
    pd.set_option('mode.chained_assignment', None)
    Y = X_train["temperature"]
    #print(Y)
    X_train.drop(columns=["temperature"],inplace = True)
    #print(X_train)
    actual = X_test["temperature"]
    X_test.drop(columns=["temperature"],inplace = True)
    p=[2,3]
    trn_bar=[]
    tst_bar=[]
    for i in p:
        print("degree of polynomial is ",i)
        poly = PolynomialFeatures(degree = i )
        x_poly = poly.fit_transform(X_train)
        test_poly = poly.fit_transform(X_test)
        poly.fit(x_poly,Y)
        linreg = li.LinearRegression()
        linreg.fit(x_poly,Y)
        pred_qual = linreg.predict(test_poly)
        pred_trn_err = math.sqrt(metrics.mean_squared_error(Y,linreg.predict(x_poly)))
        pred_tst_err = math.sqrt(metrics.mean_squared_error(actual,pred_qual))
        trn_bar.append(pred_trn_err)
        tst_bar.append(pred_tst_err)
        print("values of RMSE for degree of polynomial ",i," is ")
        print("RMSE of training data is ",pred_trn_err)
        print("RMSE of test data is ",pred_tst_err)
        plt.scatter(actual,pred_qual)
        plt.xlabel("test quality")
        plt.ylabel("pred quality")
        plt.show()
    plt.bar(p,trn_bar,color="orange")
    plt.xlabel("degree of polynomial")
    plt.ylabel("RMSE of train data")
    plt.title("bar graph for train data error")
    plt.show()
    plt.bar(p,tst_bar,color="green")
    plt.xlabel("degree of polynomial")
    plt.ylabel("RMSE of test data")
    plt.title("bar graph for test data error")
    plt.show() 
        
def corr_column(df):
    col_nm = list(df.columns)
    col_nm.remove("temperature")
    val = []
    for i in col_nm:
        val.append(abs(st.pearsonr(df["temperature"],df[i])[0]))
    x = sorted(val,reverse = True)
    col1 = col_nm[val.index(x[0])]
    col2 = col_nm[val.index(x[1])]
    return col1,col2
         
def corr_lin(df,X_train,X_test):
    col1,col2 = corr_column(df)
    X = X_train[[col1,col2]]
    Y = X_train["temperature"]
    test = X_test[[col1,col2]]
    actual = X_test["temperature"]
    reg = li.LinearRegression()
    reg.fit(X,Y)
    pred_qual = reg.predict(test)
    pred_trn_qual = reg.predict(X)
    pred_trn_err = math.sqrt(metrics.mean_squared_error(Y,reg.predict(X)))
    pred_tst_err = math.sqrt(metrics.mean_squared_error(actual,reg.predict(test)))
    print("RMSE of training data is ",pred_trn_err)
    print("RMSE of test data is ",pred_tst_err)
    plt.scatter(actual,pred_qual)
    plt.xlabel("test quality")
    plt.ylabel("pred quality")
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111 , projection = '3d')
    ax.scatter(X_train[col1],X_train[col2],Y,color="g",marker=".")
    fig1 = plt.gcf()
    fig1.set_size_inches(10, 10)
    ax.plot_trisurf(X_train[col1],X_train[col2],pred_trn_qual,alpha=0.5)
    ax.scatter(X_train[col1],X_train[col2],pred_trn_qual,color="r",marker="*")
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_zlabel("temperature")
    plt.show()

def corr_poly(df,X_train,X_test):
    col1,col2 = corr_column(df)
    X = X_train[[col1,col2]]
    Y = X_train["temperature"]
    test = X_test[[col1,col2]]
    actual = X_test["temperature"]
    RMSE_trn=[]
    RMSE_tst=[]
    p=[1,2,3,4,5]
    for i in p:
        print("degree of polynomial is ",i)
        poly = PolynomialFeatures(degree = i )
        x_poly = poly.fit_transform(X)
        test_poly = poly.fit_transform(test)
        poly.fit(x_poly,Y)
        linreg = li.LinearRegression()
        linreg.fit(x_poly,Y)
        pred_qual = linreg.predict(test_poly)
        pred_trn_err = math.sqrt(metrics.mean_squared_error(Y,linreg.predict(x_poly)))
        pred_tst_err = math.sqrt(metrics.mean_squared_error(actual,linreg.predict(test_poly)))
        RMSE_trn.append(pred_trn_err)
        RMSE_tst.append(pred_tst_err)
        print("RMSE of training data is ",pred_trn_err)
        print("RMSE of test data is ",pred_tst_err)

        plt.scatter(actual,pred_qual)
        plt.xlabel("test quality")
        plt.ylabel("pred quality")
        plt.show()
        
        sort_axis = operator.itemgetter(0)
        sorted_zip = sorted(zip(X_train[col1],X_train[col2],linreg.predict(x_poly)), key=sort_axis)
        x1,y1, y_poly_pred = zip(*sorted_zip)
        
        fig = plt.figure()
        ax = fig.add_subplot(111 , projection = '3d')
        ax.scatter(X_train[col1],X_train[col2],Y,color="g",marker=".")
        fig1 = plt.gcf()
        fig1.set_size_inches(10, 10)
        ax.plot_trisurf(x1,y1,y_poly_pred,alpha=0.5)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_zlabel("temperature")
        plt.show()

    plt.bar(p,RMSE_trn,color="orange")
    plt.xlabel("degree of polynomial")
    plt.ylabel("RMSE of train data")
    plt.title("bar graph for train data error")
    plt.show()
    plt.bar(p,RMSE_tst,color="green")
    plt.xlabel("degree of polynomial")
    plt.ylabel("RMSE of test data")
    plt.title("bar graph for test data error")
    plt.show()    
    


    
def main():
    path = "atmosphere_data.csv"
    df = readfile(path)
    X_train,X_test = trn_tst(df)
    #linear_reg(X_train,X_test) #ques1
    #poly_reg(X_train,X_test) #ques2
    #mul_lin_reg(X_train,X_test)
    #mul_poly_reg(X_train,X_test)
    #corr_column(df)
    corr_lin(df,X_train,X_test) #ques3
    #corr_poly(df,X_train,X_test) 
main()