from numpy import array as ar
import numpy as np
from math import pi
import matplotlib.pyplot as plt

t = np.arange(0,10*pi,0.01)

y1 = 10*np.sin(t)
y2 = np.sin(10*t)

y = y1 + y2

plt.plot(t,y)
plt.show()
