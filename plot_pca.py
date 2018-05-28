import numpy as np
import matplotlib.pyplot as plt
import pca

def plot_eigenvectors(p):
    for vec in p.E.T:
        d = vec.reshape(28,28) * (1/np.amax(vec))
        plt.imshow(d, interpolation='nearest', cmap='gray')
        plt.show()

p = pca.pca('data-1.txt')
y = p.data.dot(p.E)
plt.imshow(p.mean.reshape(28,28), cmap='gray', interpolation='nearest', vmin=0, vmax=255)
plt.show()
plot_eigenvectors(p)
print(np.argmax(y, axis=0))
for i in np.argmax(y, axis=0):
    plt.imshow(p.data[i].reshape(28,28), cmap='gray', vmin=0, vmax=255)
    plt.show()
