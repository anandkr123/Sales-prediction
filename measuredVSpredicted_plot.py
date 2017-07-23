import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
data=pd.read_excel('features.xlsx')
X=data[['DAY','TEMP','CLOUD','RAIN','WIND','SCHD','PUHD','PRSA','Seasons']];
y=data['Sales']
lr=LinearRegression();

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation (Data is predicted more variantly as it is splitted into folds)
predicted = cross_val_predict(lr, X, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
