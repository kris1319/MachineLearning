import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = np.genfromtxt('perceptron-train.csv', delimiter=',')
y_train = data[:, [0]].transpose()[0]
X_train = data[:, 1:]
data = np.genfromtxt('perceptron-test.csv', delimiter=',')
y_test = data[:, [0]].transpose()[0]
X_test = data[:, 1:]

perc = Perceptron(random_state=241)
perc.fit(X_train, y_train)
acc1 = accuracy_score(y_test, perc.predict(X_test))

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
perc.fit(X_train, y_train)
acc2 = accuracy_score(y_test, perc.predict(X_test))

print(acc1, acc2)
print(acc2 - acc1)
