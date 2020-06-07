import pandas as pd
import numpy as np
from glosys_ml.accuracy import Accuracy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)

class Models:
	
	def logistic_regression(x,y):
		x_train,x_test,y_train,y_test=train_test_split(x, y, test_size = 0.2, random_state = 0)
		classifier=LogisticRegression()
		classifier.fit(x_train,y_train)
		Accuracy.finding_accuracy(x_train,y_train,x_test,y_test,classifier)

	def linear_regression(x,y):
		x_train,x_test,y_train,y_test=train_test_split(x, y, test_size = 0.2, random_state = 0)
		classifier=LinearRegression()
		classifier.fit(x_train,y_train)
		Accuracy.finding_accuracy(x_train,y_train,x_test,y_test,classifier)

	def naive_bayes(x,y):
		x_train,x_test,y_train,y_test=train_test_split(x, y, test_size = 0.2, random_state = 0)
		classifier = GaussianNB()
		classifier.fit(x_train,y_train)
		Accuracy.finding_accuracy(x_train,y_train,x_test,y_test,classifier)

	def decision_tree(x,y):
		x_train,x_test,y_train,y_test=train_test_split(x, y, test_size = 0.2, random_state = 0)
		classifier=DecisionTreeClassifier()
		classifier.fit(x_train,y_train)
		Accuracy.finding_accuracy(x_train,y_train,x_test,y_test,classifier)

	def neural_network(x,y):
		x_train,x_test,y_train,y_test=train_test_split(x, y, test_size = 0.2, random_state = 0)
		classifier=MLPClassifier()
		# classifier=MLPClassifier(hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, batch_size=’auto’, learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)
		classifier.fit(x_train,y_train)
		Accuracy.finding_accuracy(x_train,y_train,x_test,y_test,classifier)

	def support_vector_machine(x,y):
		x_train,x_test,y_train,y_test=train_test_split(x, y, test_size = 0.2, random_state = 0)
		classifier = SVC(kernel = 'rbf')
		classifier.fit(x_train, y_train)
		Accuracy.finding_accuracy(x_train,y_train,x_test,y_test,classifier)

	def random_forest(x,y):
		x_train,x_test,y_train,y_test=train_test_split(x, y, test_size = 0.2, random_state = 0)
		classifier = RandomForestClassifier(n_estimators = 10)
		classifier.fit(x_train, y_train)
		Accuracy.finding_accuracy(x_train,y_train,x_test,y_test,classifier)