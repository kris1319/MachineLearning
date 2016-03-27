import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space'])
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(newsgroups.data)
y = newsgroups.target

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
svc = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(svc, grid, scoring='accuracy', cv=cv)

gs.fit(X, y)
opt_C, opt_acc = 1e+6, 0
for i in gs.grid_scores_:
	print(i.mean_validation_score, i.parameters['C'])
	if i.mean_validation_score > opt_acc or (i.mean_validation_score == opt_acc and i.parameters['C'] < opt_C):
		opt_C = i.parameters['C']
		opt_acc = i.mean_validation_score

print(opt_C, opt_acc)
svc = SVC(C=opt_C, kernel='linear', random_state=241)
svc.fit(X, y)
features = tfidf.get_feature_names()
arr = list()
for i in range(svc.coef_.shape[1]):
	arr.append((i, svc.coef_[0, i]))
arr.sort(key=lambda x: abs(x[1]), reverse=True)
for i in range(10):
	print(features[arr[i][0]])
