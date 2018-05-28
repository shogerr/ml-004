import numpy as np

class kmean:
    def __init__(self, data, k=2):
        self.k = k 
        # ingest data
        self.data = data * (1/255)
        # select k random centroids
        self.c = self.data[np.random.randint(self.data.shape[0], size=k), :]
        # perform initial classification
        self.group()

    def find_cost(self, x, i):
        return np.linalg.norm(x-self.c[self.groups[i],:])**2
    # determine the current loss value
    #
    def cost(self):
        return np.array([self.find_cost(v,i) for i,v in enumerate(self.data)]).sum()

    # Takes a matrix s of all examples in a group and calculates
    # a centroid vector.
    def find_centroid(self, s):
        # If the cluster is empty, supply a random example as the centroid
        if s.shape[0] == 0:
            return self.data[np.random.randint(self.data.shape[0],size=1), :]

        return s.sum(axis=0)*(1/s.shape[0])

    # Takes a feature vector and finds the closest cluster centroid.
    def find_group(self, x):
        d_min = float('inf')
        i = None
        j = 0
        for m in self.c:
            d = np.linalg.norm(x-m)
            if d < d_min: 
                d_min = d
                i = j
            j+=1

        return i

    def update(self):
        for i in range(self.k):
            self.c[i] = self.find_centroid(self.data[np.where(self.groups == i)[0],:])

    def group(self):
        self.groups = np.apply_along_axis(self.find_group, axis=1, arr=self.data)

    def run(self):
        results = []
        for i in range(1, 100):
            self.update()
            self.group()
            results.append(self.cost())
            if len(results) > 2 and results[-2] - results[-1] < .0001:
                break

        return results
