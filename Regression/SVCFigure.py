# =============================================================================
# SVM FIGURE
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import KernelPCA
f = lambda x: 2 * x - 5
f_upp = lambda x: 2 * x - 5 + 2
f_lower = lambda x: 2 * x - 5 - 2

pos = []
neg = []

np.random.seed(345234)
for i in range(80):
    x = np.random.randint(15)
    y = np.random.randint(15)

    d = np.abs(2*x-y-5)/np.sqrt(2**2+1)
    if f(x) < y and d>=1:
        pos.append([x,y])
    elif f(x) > y and d>=1 :
        neg.append([x,y])

pos.append([4, f_upp(4)])
neg.append([8, f_lower(8)])


plt.figure()
plt.xticks([])
plt.yticks([])
plt.scatter(*zip(*pos))
plt.scatter(*zip(*neg))

plt.plot([0,10],[f(0),f(10)], linestyle='--', color='m')
plt.plot([0,10],[f_upp(0),f_upp(10)], linestyle='--', color='red')
plt.plot([0,10],[f_lower(0),f_lower(10)], linestyle='--', color='red')
plt.plot([4,3],[f_lower(4),f_upp(3)], linestyle='-', color='black')
plt.plot([7,6],[f_lower(7),f_upp(6)], linestyle='-', color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.title('SVM')
