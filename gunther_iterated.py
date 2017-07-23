import pandas as pd
import sys, getopt
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data=pd.read_excel('features.xlsx')


feature_cols=['DAY','TEMP','CLOUD','RAIN','WIND','PUHD','SCHD','PRSA','Seasons']
X=data[feature_cols];
y=data['Sales']
y_gunther=data['PRSA']          # PREDICTIVE SALES OF GUNTHER
lr=LinearRegression()


min_rmse=10000
lr_opt=LinearRegression()

print("\n=======Iterating for 15 times splitting randomly to find the best fitting LR model ==========")
for i in range(1,15):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8)
    lr.fit(X_train,y_train)
    y_pred=lr.predict(X_test)
    rmse=np.sqrt(metrics.mean_squared_error(y_test,y_pred))
    if min_rmse > rmse:
        min_rmse=rmse
        lr_opt=lr
    

print("\nOur training intercept ",lr_opt.intercept_)
print("\nOur training coefficients\n")
print(feature_cols)
print(lr_opt.coef_)
print("\nThe minimum root mean square error over the iteration is \n",min_rmse)
print("==============fitting the same model for overall price predictions=============")
y_pred_all=lr_opt.predict(X)
gunther_loss = y_gunther.sum()-y.sum()
our_pred=y_pred_all.sum()-y.sum()
print("\nGunter loss from Jan 2016 to April 2017 compared to actual sales ",gunther_loss)
if our_pred <0:
    print("\nOur prediction is only ",-our_pred," less than actual sales")
else:
    print("\n Our predition is only ",our_pred," more than actual sales",)

print("\n=========== predicting the price =========== \n")
x1=float(sys.argv[1])
x2=float(sys.argv[2])
x3=float(sys.argv[3])
x4=float(sys.argv[4])
x5=float(sys.argv[5])
x6=float(sys.argv[6])
x7=float(sys.argv[7])
x8=float(sys.argv[9])
x9=float(sys.argv[9])
X_enter=[x1,x2,x3,x4,x5,x6,x7,x8,x9]
y_new_pred=lr_opt.predict(X_enter)

print("\nThe predicting price for the day is ",y_new_pred)
