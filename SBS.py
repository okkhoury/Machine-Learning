#Sequential Backward Selection
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from itertools import combinations
import numpy as np 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap


class SBS():
	def __init__(self, estimator, k_features, scoring = accuracy_score,
			test_size = 0.25, random_state=1):
		self.scoring = scoring
		self.estimator = clone(estimator)
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state

	def fit(self, X, y):
		X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

		dim = X_train.shape[1]
		self.indices_ = tuple(range(dim))
		self.subsets_ = [self.indices_]
		score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)

		self.scores_ = [score]

		while dim > self.k_features:
			scores = []
			subsets = []

			for p in combinations(self.indices_, r=dim-1):
				score = self._calc_score(X_train, y_train, X_test, y_test, p)
				scores.append(score)
				subsets.append(p)

			best = np.argmax(scores)
			self.indices_ = subsets[best]
			self.subsets_.append(self.indices_)
			dim -= 1

			self.scores_.append(scores[best])
		self.k_score_ = self.scores_[-1]

		return self


	def transform(self, X):
		return X[:, self.indices_]

	def _calc_score(self, X_train, y_train, X_test, y_test, indices):
		self.estimator.fit(X_train[:, indices], y_train)
		y_pred = self.estimator.predict(X_test[:, indices])
		score = self.scoring(y_test, y_pred)
		return score


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
	#setup marker generator and color map
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'grey', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	#plot the decision surface
	# X[:,0] = take all rows(:), only take the first column(0)
	x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)

	plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	#plot all samples
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

	#highlight test samples
	if test_idx:
		X_test, y_test = X[test_idx, :], y[test_idx]
		plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidths=1, marker='o', s=55, label='test set')

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
					'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 
					'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/0D315 of diluted wines', 'Proline']
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)


knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
#plt.show()

k5 = list(sbs.subsets_[8])
#print(df_wine.columns[1:][k5])

# knn.fit(X_train_std, y_train)
# print('Training accuracy:', knn.score(X_train_std, y_train))
# print('Testing accuracy:', knn.score(X_test_std, y_test))

print()

knn.fit(X_train_std[:, k5], y_train)
#print('Training accuracy:', knn.score(X_train_std[:, k5], y_train))
# the book gets a testing accuracy of 96 percent. I'm only getting 92. Don't know why
#print('Testing accuracy:', knn.score(X_test_std[:, k5], y_test))


'''Determining feature importance by using a random forest'''
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
# for f in range(X_train.shape[1]):
# 	print("%2d) %-*s %f" % (f+1, 30, feat_labels[indices[f]], importances[indices[f]]))

X_selected = forest.transform(X_train, threshold=0.15)
#print(X_selected.shape)


'''LDA feature extraction using sklearn'''
from sklearn.lda import LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel("LD 2")
plt.legend(loc='lower left')
plt.show()








