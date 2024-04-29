import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

class LinearRegression(object):
    
    def __init__(self, _input, _output):
        self._input = _input
        self._output = _output

    def fit(self):

        n = np.size(self._input)
        m_x, m_y = np.mean(self._input), np.mean(self._output)

        SS_xy = np.sum(self._output*self._input) - n*m_y*m_x
        SS_xx = np.sum(self._input*self._input) - n*m_x*m_x

        self.b_1 = SS_xy / SS_xx
        self.b_0 = m_y - self.b_1*m_x
        print(self.b_0,self.b_1)

    def predict(self):
        plt.scatter(self._input, self._output, color = "m", 
               marker = "o", s = 30) 
        y_pred =self.b_0 +self.b_1*self._input
        plt.plot(self._input, y_pred, color = "g")
        plt.show()


p=LinearRegression(x,y)   
p.fit()
p.predict()
