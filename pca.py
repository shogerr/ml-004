import numpy as np
from heapq import nlargest

class pca:
	# data_path: string of data file path
	def __init__(self, data_path):
		# load data from file
		self.data = np.loadtxt(data_path, dtype=np.int, delimiter=',')

		# calculate mean
		self.mean = np.mean(self.data)

		# calculate covariance matrix
		self.covariance = np.cov(self.data)

		# calculate eigen-values
		# np.linalg.eig returns values in decending order
		self.eigen_vals, self.eigen_vecs = np.linalg.eig(self.covariance)

		# get top 10 eigen vectors (first 10)
		self.top_eigen_vecs = self.eigen_vecs[:10]
