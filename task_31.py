import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = np.genfromtxt('svm-data.csv', delimiter=',')
y = data[:, [0]].transpose()[0]
X = data[:, 1:]

svc = SVC(C=100000, kernel='linear', random_state=241)
svc.fit(X, y)

print(svc.support_)