import pandas
import numpy

data = pandas.read_csv('titanic.csv', usecols=['Pclass', 'Sex', 'Age', 'Fare', 'Survived']).dropna(how='any')
data.Sex = data.Sex.map({'female' : 0, 'male' : 1})
X = numpy.array(data.loc[:, ['Pclass', 'Sex', 'Age', 'Fare']].values)
y = data.Survived.values

from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state=241)
#['Pclass', 'Sex', 'Age', 'Fare']
clf = clf.fit(X, y)

print(clf.feature_importances_)