# credits: https://www.youtube.com/watch?v=8UC_GaY9U1U&list=PLKWX1jIoUZaVpVhMfevAE7iYNcDHPEJI_&index=37

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
gap = e.T @ X @ Z @ e

def f_dPi(A,Z,X,b,mu,e,f_pi):

    inv(np.dot())




# for i in range(1,8):  # test

while gap > threshold:

    print(f'\n\n Iteration:{counter} with Gap = {gap} \n')

    delta_dual_feasibility = A.T @ dPi - dZ
    print(f'\n Delta Dual Feasibility {delta_dual_feasibility}')

    dPi = inv( A @ inv(Z) @ X @ A.T) @ (-b + mu * (A @ inv(Z) @ e) + A @ inv(Z) @ X @ delta_dual_feasibility)
    print(f'Delta Pi {dPi}')

    dZ = -delta_dual_feasibility + A.T @ dPi
    print(f'Delta Z {dZ}')

    dX = inv(Z) @ (mu * e - X @ Z @ e - X @ dZ)
    print(f'Delta X {dX}')

    # Atualizar os valores de X, Pi e Z de acorodo com uma taxa de atualização em cada interação
    # A Direcão será de acordo com o vetor gradient de cada variável
    X = X + eta * np.diag(dX)
    print(f'\n Valor de X atualizado: {np.diag(X)}')

    Pi = Pi + eta * np.diag(dPi)
    print(f'Valor de Pi atualizado: {np.diag(Pi)}')

    Z = Z + eta * np.diag(dZ)
    print(f'Valor de Z atualizado: {np.diag(Z)}')

    # Atualizar o valor de gap, dado os novos valores de X, Pi e Z 
    # porque essa multiplicacao ?????
    gap = e.T @ X @ Z @ e 
    print(f'\n Valor de gap atualizado {gap}')

    # Conforme fica mais perto da barreira, diminuir o valor de mu
    mu = gap/(counter**2)
    print(f'Valor de mu atualizado {mu}')

    counter += 1
    





















































