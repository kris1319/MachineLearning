import numpy as np
import pandas
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

def preprocess(name):
	data = pandas.read_csv(name, delimiter=',')
	data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True, inplace=True)
	#data['FullDescription'] = data['FullDescription'].apply(str.lower)
	data['LocationNormalized'].fillna('nan', inplace=True)
	data['ContractTime'].fillna('nan', inplace=True)

	return data

data_train = preprocess('salary-train.csv')
data_test = preprocess('salary-test-mini.csv')

tfidf = TfidfVectorizer(min_df=5, lowercase=True)
Ft_train = tfidf.fit_transform(data_train.FullDescription)

one_hot = DictVectorizer()
Ctg_train = one_hot.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_train = hstack([Ft_train, Ctg_train])
est = Ridge(alpha=1, random_state=241)
est.fit(X_train, data_train.SalaryNormalized)

Ctg_test = one_hot.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
Ft_test = tfidf.transform(data_test.FullDescription)
X_test = hstack([Ft_test, Ctg_test])
print(est.predict(X_test))