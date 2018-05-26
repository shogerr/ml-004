import kmean
import numpy as np

examples = np.loadtxt('data-1.txt', dtype=np.int, delimiter=',')

# Run with k=2
model = kmean.kmean(examples)

results = model.run()
np.savetxt('kmean_k_2.csv', results)

results = []
# Explore different values of k
for k in range(2,11):
    r_min = float('inf')
    # Run each k 10 times, pick the smallest cost
    for i in range(10):
        model = kmean.kmean(examples, k=k)
        r = model.run()
        if r[-1] < r_min:
            r_min = r[-1]

    results.append(r_min)

output = np.column_stack((np.arange(2,11),results))
np.savetxt('kmean_all_k.csv', output)
print(output)
