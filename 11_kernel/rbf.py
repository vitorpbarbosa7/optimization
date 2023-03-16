import numpy as np
from numpy import array as ar
from math import exp
import matplotlib.pyplot as plt

def rbf(xn:np.array, xm:np.array, sigma:float):

	l2_norm = (xn - xm)**2
	variance = sigma**2

	gamma = 1/variance
	print(gamma)
	
	k_xn_xm = exp((-l2_norm)/(2*variance))

	return k_xn_xm


xn = ar([2.5])
xm = ar([4])


sigmas = np.arange(start = 0.01, stop = 10, step = 0.05)

res_gammas = [rbf(xn, xm, sigma) for sigma in sigmas]


plt.scatter(x = sigmas, y = res_gammas)
plt.show()

