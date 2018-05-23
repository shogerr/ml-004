import numpy as np

class kmean:
    def __init__(self, data, k=2):
        self.k = k 
        # ingest data
        self.data = data
        # select k random centroids
        self.c = data[np.random.randint(data.shape[0], size=k), :].astype(np.float)
        # perform initial classification
        self.group()

    # determine the current loss value
    def cost(self):
        pass

    # Takes a matrix s of all examples in a group and calculates
    # a centroid vector.
    def find_centroid(self, s):
        return s.sum(axis=0)*(1/s.shape[0])

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
        self.update()

    def train(self):
        pass
        # group each example

