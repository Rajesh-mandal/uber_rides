import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import math

data = pd.read_csv('taxi.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

print(data.describe())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 0 )

from sklearn.linear_model import LinearRegression
lreg = LinearRegression()
lreg.fit(X_train,y_train)
lreg.score(X_train,y_train)

#predict the output
y_pred = lreg.predict(X_test)

#r2_score
from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)

#calculate mse 
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)

#calculate rmse
rmse = math.sqrt(mse)

#perform cross validate
from sklearn.model_selection import cross_validate
scores = cross_validate(lreg, X, y, cv=3,scoring=('r2', 'neg_mean_squared_error'),
return_train_score=True)
sc1=scores['test_neg_mean_squared_error']
sc2 = scores['train_r2'].mean()

pickle.dump(lreg, open('taxi.pkl','wb')) #dump the lreg in taxi.pkl
model = pickle.load(open('taxi.pkl','rb'))  #load the pickle file

print(model.predict([[80,1770000,6000,85]]))