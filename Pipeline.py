import pandas as pd 

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

from sklearn.preprocessing import LabelEncoder
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Use a pipline to put all of teh transofrmations together
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline 
pipe_lr = Pipeline([('scl', StandardScaler()), 
					('pca', PCA(n_components=2)),
					('clf', LogisticRegression(random_state=1))])
pipe_lr.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

# Use K-fold cross validation to test the model
import numpy as np 
from sklearn.cross_validation import StratifiedKFold

kfold = StratifiedKFold(y=y_train, n_folds=10, random_state=1)
scores = []
# for k, (train, test) in enumerate(kfold):
# 	pipe_lr.fit(X_train[train], y_train[train])
# 	score  = pipe_lr.score(X_train[test], y_train[test])
# 	scores.append(score)

#print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# Finding the best hyperparameters via grid search
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC 
from sklearn.cross_validation import cross_val_score
pipe_svc = Pipeline([('scl', StandardScaler()),
					 ('clf', SVC(random_state=1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
			  {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
# gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10)
# gs = gs.fit(X_train, y_train)
# print(gs.best_score_)
# print(gs.best_params_)

#clf = gs.best_estimator_
#clf.fit(X_train, y_train)

gs = GridSearchCV(estimator=pipe_svc,
		param_grid = param_grid,
		scoring='accuracy',
		cv=2,)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(
		estimator=DecisionTreeClassifier(random_state=1),
		param_grid=[
			{'max_depth': [1,2,3,4,5,6,7,None]}],
			scoring='accuracy',
			cv=5)

scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=2)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))









