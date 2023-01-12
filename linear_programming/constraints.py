import matplotlib.pyplot as plt

def constr_1(x):
	
	return x/2 + 4

superior = [constr_1(x) for x in range(1,20)]


plt.plot(superior)
plt.show()
