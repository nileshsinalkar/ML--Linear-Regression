# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 00:26:01 2019

@author: niles
"""

import numpy as np
import matplotlib.pyplot as plt
import  pandas as pd

dataset= pd.read_csv('Salary_data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=3)

from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train,y_train)

pred=regression.predict(x_test)

plt.scatter(x, y, color = 'red')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_train,y_train,color='green')
plt.plot(x_train,regression.predict(x_train),color='yellow')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test,y_test,color='green')
plt.plot(x_test,regression.predict(x_test),color='yellow')
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test,y_test,color='green')
plt.plot(x_test,pred,color='yellow')
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

from sklearn.metrics import mean_squared_error
ni=mean_squared_error(y_test, pred)

p=regression.predict([[2.9],[3.8]])

r=regression.predict([[2.9],[3.8]])

