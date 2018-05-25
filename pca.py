import numpy as np

class pca:
	def __init__(self, data_path):
		self.data = np.loadtxt(data_path, dtype=np.int, delimiter=',')
