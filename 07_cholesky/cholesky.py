import numpy as np

G = [[5,3,0,1,2],[3,4,-2,4.5,0],[0,-2,3,-3,3],[1,4.5,-3,2,2.6],[2,0,3,2.6,5]]

eigvals = np.linalg.eigvals(G)
print(eigvals)

v, _ = np.linalg.eig(G)
print(v)
