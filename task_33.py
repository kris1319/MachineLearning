import numpy as np
import math
from sklearn.metrics import roc_auc_score

data = np.genfromtxt('data-logistic.csv', delimiter=',')
y = data[:, [0]].transpose()[0]
X = data[:, 1:]
l = len(y)

def grad_search(w, step, C):
	for i in range(10000):
		sum0, sum1 = 0, 0
		for j in range(l):
			k = 1 + math.exp(y[j] * (w[0] * X[j, 0] + w[1] * X[j, 1]))
			sum0 += y[j] * X[j, 0] / k
			sum1 += y[j] * X[j, 1] / k

		w0 = w[0] + step * (sum0 / l - C * w[0])
		w1 = w[1] + step * (sum1 / l - C * w[1])

		if math.sqrt(math.pow(w0 - w[0], 2) + math.pow(w1 - w[1], 2)) < 1e-5:
			print('iter  ', i)
			return (w0, w1)
		w = (w0, w1)

	print('iter  ', i)
	return w

w = grad_search((0, 0), 0.001, 0)
pw = [ 1 / (1 + math.exp(- w[0] * X[j, 0] - w[1] * X[j, 1])) for j in range(l)]
print(roc_auc_score(y, pw))
w = grad_search((0, 0), 0.001, 4)
pw = [ 1 / (1 + math.exp(- w[0] * X[j, 0] - w[1] * X[j, 1])) for j in range(l)]
print(roc_auc_score(y, pw))