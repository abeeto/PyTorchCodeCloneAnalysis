# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:19:43 2022

@author: Mahfuz Shazol
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,10*np.pi,1000)

y=np.sin(x)

plt.plot(x, y)
plt.show()
