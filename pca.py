import numpy as np

class pca:
	# data_path: string of data file path
	def __init__(self, data_path):
		# load data from file
		self.data = np.loadtxt(data_path, dtype=np.int, delimiter=',')

		# calculate mean
		self.mean = np.mean(self.data)

		# calculate covariance matrix
		self.covariance = np.cov(self.data)
