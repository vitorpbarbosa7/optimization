import numpy as np
from scipy.optimize import minimize

def objective(x):
	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]
	return x1*x4*(x1+x2+x3)+x3

def constraint_1(x):
	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]

	return x1*x2*x3*x4 - 25

def constraint_2(x):
	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]

	return (x1**2) + (x2**2) + (x3**2) + (x4**2) - 40

# initial guess
x0 = (1,5,5,1)

print(objective(x0))


# bounds
bnd = (1,5)
bounds = (bnd,bnd,bnd,bnd)

con1 = {'type':'ineq', 'fun':constraint_1}
con2 = {'type':'eq', 'fun':constraint_2}
cons = [con1,con2]

solution = minimize(fun = objective, x0 = x0, bounds = bounds, constraints = cons, method='SLSQP')

print(solution)
#print(solution.res)
