import numpy as np
import pandas
from sklearn import metrics

data = np.genfromtxt('classification.csv', delimiter=',')
y_true = data[:, [0]].transpose()[0][1:]
y_pred = data[:, [1]].transpose()[0][1:]
l = len(y_true)

# mistake table
tp, tn, fp, fn = 0, 0, 0, 0
for i in range(l):
	if y_true[i] == 0:
		if y_true[i] == y_pred[i]:
			tn += 1
		else:
			fp += 1
	else:
		if y_true[i] == y_pred[i]:
			tp += 1
		else:
			fn += 1
print(tp, fp, fn, tn)

# metrics
print(metrics.accuracy_score(y_true, y_pred))
print(metrics.precision_score(y_true, y_pred))
print(metrics.recall_score(y_true, y_pred))
print(metrics.f1_score(y_true, y_pred))

# auc-roc for scores.csv
df = pandas.read_csv('scores.csv')
score_logreg = df['score_logreg'].tolist()
score_svm = df['score_svm'].tolist()
score_knn = df['score_knn'].tolist()
score_tree = df['score_tree'].tolist()
yt = df['true'].tolist()
print('logreg', metrics.roc_auc_score(yt, score_logreg))
print('svm', metrics.roc_auc_score(yt, score_svm))
print('knn', metrics.roc_auc_score(yt, score_knn))
print('tree', metrics.roc_auc_score(yt, score_tree))

def precision(est, y, pred):
	maxprec = 0
	prec, rec, thresh = metrics.precision_recall_curve(y, pred)
	for i in range(rec.shape[0]):
		if rec[i] >= 0.7 and prec[i] > maxprec:
			maxprec = prec[i]
	print(est, maxprec)

precision('log', yt, score_logreg)
precision('svm', yt, score_svm)
precision('knn', yt, score_knn)
precision('tree', yt, score_tree)