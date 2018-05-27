import numpy as np
from heapq import nlargest

class pca:

	# data_path: string of data file path
	def __init__(self, data_path, data_type=np.int):
		# load data from file
		self.data = np.loadtxt(data_path, dtype=data_type, delimiter=',')

		# calculate mean
		self.mean = np.mean(self.data, axis=0)

		# calculate covariance matrix
		self.covariance = np.cov(self.data, rowvar=False)

		# calculate eigen-values
		self.eigen_vals, self.eigen_vecs = np.linalg.eigh(self.covariance)

		# get top 10 eigen vectors
		largest_indexs = nlargest(10, range(len(self.eigen_vals)), self.eigen_vals.take)
		self.top_eigen_vecs = []
		for index in largest_indexs:
			self.top_eigen_vecs.append(self.eigen_vecs[index])
		self.top_eigen_vecs = np.array(self.top_eigen_vecs)

		# find top 10 eigenvectors which only have a single entry in them
		#count = 0
		#self.top_eigen_vecs = []
		#self.dimension_indexs = []
		#for vec in reversed(self.eigen_vecs):
			#added = 0
			#for num in vec:
				#if num != 1:
					#added += num
			#if added == 0 and len(self.top_eigen_vecs) < 10:
				#self.top_eigen_vecs.append(vec)
				#self.dimension_indexs.append(count)
			#count += 1

		# draw top 10 eigen vecs
		#for vec in self.top_eigen_vecs:
			#scaled_vec = [x/max(vec) for x in vec]
			#pyplot.imshow(np.reshape(scaled_vec,(28,28)))
			#pyplot.show()
			#input("next")

		# draw mean vector
		#pyplot.imshow(np.reshape(self.mean,(28,28)))
		#pyplot.show()

		values = self.find_highest_dimensions()
		#for i in range(len(values)):
			#print('dimension:', i, 'data vector:', values[i][1], 'value:', values[i][0])

		# get list of vectors with highest values after
		# eigenvector reduction
		self.highest_dim_vecs = [self.data[x[1]] for x in values]

		# draw images reduced by eigenvectors
		# only draws images with highest value
		# for one of the ten dimensions
		for vec in self.highest_dim_vecs:
			pyplot.imshow(np.reshape(np.dot(self.top_eigen_vecs, vec),(5,2)))
			pyplot.show()
			input("next")
			pyplot.imshow(np.reshape(vec,(28,28)))
			pyplot.show()
			input("next")

	# unused
	# calculate mean of variables (columns)
	def calc_variable_mean(self):
		self.mean = []
		# transpose data so for in loop uses columns instead of rows
		for column in self.data.T:
			# add up each number in row and divide by length
			column_mean = 0
			for num in column:
				column_mean += num
			column_mean /= len(column)
			self.mean.append(column_mean)

	# finds vector from original data set with highest values
	# in top ten eigenvector dimensions
	def find_highest_dimensions(self):
		values = []
		for i in range(10):
			values.append((0, 0))
		for i in range(len(self.data)):
			vec = np.dot(self.top_eigen_vecs, self.data[i])
			for j in range(len(vec)):
				if vec[j] > values[j][0]:
					values[j] = (vec[j], i)
		return values
