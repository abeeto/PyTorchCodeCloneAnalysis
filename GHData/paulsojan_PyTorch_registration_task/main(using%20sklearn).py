import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model

x = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
xTrain, xTest, yTrain, yTest = train_test_split(
    x, y, test_size=0.2, random_state=0)

regr = LinearRegression()
class LinearRegression(object):

  def __init__(self,_input,_output):
    self._input=_input
    self._output=_output

  def fit(self):
    regr.fit(self._input,self._output)


  def predict(self):
      y_pred = regr.predict(xTest)
      print(y_pred)
      plt.scatter(xTest, yTest, color ='b') 
      plt.plot(xTest,y_pred, color ='k')  
      plt.show() 
      


p=LinearRegression(xTrain,yTrain)
p.fit()
p.predict()
