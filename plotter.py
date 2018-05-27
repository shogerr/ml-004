import numpy as np
from matplotlib import pyplot

def pca_plotter(pca):
	# draw top 10 eigen vecs
	#for vec in pca.top_eigen_vecs:
		#scaled_vec = [x/max(vec) for x in vec]
		#pyplot.imshow(np.reshape(scaled_vec,(28,28)))
		#pyplot.show()

	# draw mean vector
	#pyplot.imshow(np.reshape(pca.mean,(28,28)))
	#pyplot.show()

	# draw images reduced by eigenvectors
	# only draws images with highest value
	# for one of the ten dimensions
	for vec in pca.highest_dim_vecs:
		pyplot.imshow(np.reshape(np.dot(pca.top_eigen_vecs, vec),(5,2)))
		pyplot.show()
		pyplot.imshow(np.reshape(vec,(28,28)))
		pyplot.show()
