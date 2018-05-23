import kmean
import numpy as np

examples = np.loadtxt('unsupervised.txt', dtype=np.int, delimiter=',')

model = kmean.kmean(examples)

model.run()
