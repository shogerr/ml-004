import matplotlib.pyplot as plt
import numpy as np

d = np.loadtxt('kmean_k_2.csv').astype('float')

fig, ax = plt.subplots()

ax.plot(np.arange(1,len(d)+1), d)

ax.set_ylabel('cost')
ax.set_xlabel('iterations')
ax.set_title('Cost convergence at k=2')
fig.savefig('kmeans_k_2.png')

d = np.loadtxt('kmean_all_k.csv').astype('float')

fig, ax = plt.subplots()

ax.plot(d[:,0], d[:,1])

ax.set_ylabel('cost')
ax.set_xlabel('k')
ax.set_title('Cost for range of k')
fig.savefig('kmeans_k_range.png')
