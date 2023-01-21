import numpy as np
from math import log
from numpy import array as ar
from numpy.linalg import inv

# >>>> Primal Form <<<<<

# 2*x1 + 3*x2 + 0*x3 + 0*x4
primal_obj = ar([2,3,0,0])

# >> Constraints from Primal Form
# (I) 2*x1 + x2 + x3 + 0*x4= 8
# (II) x1 + 2*x2 + 0*x3 + x4 = 6
# x1, x2, x3, x4 >= 0

A = ar([[2,1,1,0],[1,2,0,1]])

# > b terms
b = ar([8,6])

# >> Initialization for x1, x2, x3 and x4 <<

# (i) x1 = 1
# (ii) x2 = 1
# Applying (i) and (ii) to (I)
# x3 = 5
# Applying (i) and (ii) to (II)
# x4 = 3

X = np.diag([1,1,5,3])

# >>>> Dual Form <<<<

dual_obj = ar([0,0,8,6])

# >> Constraints Dual Form:
# (III) 2*z3 + z4 - z1 = 2
# (IV) z3 + 2*z4 - z2 = 3
# z1, z2, z3, z4 >= 0

# (iii) pi_2 = 2 (z3)
# (iv) pi_1 = 2 (z4)
Pi = ar([2,2])

# (iii) and (iv) to (III)
# z1 = 4
# z2 = 3

Z = np.diag([4,3,2,2])

# > Barrier mu term ????
mu = 1.4375

# Column vector for helping with matrix operations
e = np.ones(4)

# Rate of change between interations
eta = 0.995

# Vectors Delta_x, Delta_pi, Delta_z. 
dX = np.zeros(4)
dPi = np.zeros(2)
dZ = np.zeros(4)

# classic counter for iterations
counter = 1

# Stopping criteria according to the gap between primal an dual space solutions
threshold = 1E-5

# Compute initial gap 
# @ for matrix multiplication (for 2d matrix also works with np.dot and numpy.matmul)
gap = e.T @ X @ Z @ e

print(gap)