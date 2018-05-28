import numpy as np

class pca:
    def __init__(self, data_path):
        # Load examples
        self.data = np.loadtxt(data_path, dtype=np.float, delimiter=',')
        self.data = self.data
        # Find the mean matrix
        self.mean = self.data.mean(axis=0)
        # Find the covariance matrix
        self.covariance = np.cov(self.data, rowvar=False)
        V, E = np.linalg.eigh(self.covariance)
        # Sort the eigenvalues
        idx = np.argsort(V)[::-1]
        # Order the eigenvectors
        E = E[:,idx]
        self.V = V[idx]
        # Select the 10 eigenvectors with the greatset variance.
        self.E = E[:, :10]

if __name__ == '__main__':
    p = pca('data-1.txt')
    print('Ten greatest eigenvalues:')
    print(p.V[:10])
    # Perform a projection
    y = p.data.dot(p.E)

    print('Ten images with greatest value in projection dimensions:')
    print(np.argmax(y, axis=0))
