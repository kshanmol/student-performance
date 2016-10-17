import get_data
import random
import sys
import os

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import numpy as np
import matplotlib.pyplot as plt

CONST_RANDOM_SEED = 14

def reduction(x):
	return x

def svm(dataset_file_name, train_data_perc, shuffle = True, do_PCA = False, PCA_threshold = 0.99, random_seed = CONST_RANDOM_SEED):
	x, y = get_data.process(dataset_file_name, shuffle)

	train_data_length = train_data_perc * len(x) / 100
	x_train, y_train = x[:train_data_length], y[:train_data_length]
	x_test, y_test = x[train_data_length:], y[train_data_length:]
	
	if(do_PCA):
		x_train, x_test = get_data.reduced_features(x_train, x_test, PCA_threshold)
	
	# to normalize the data attributes
	scaler = StandardScaler(); scaler.fit(x_train)
	
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)

	param_grid = [
		{'C': map(lambda x: 2 ** x, range(5, 10)), 'kernel': ['rbf'], 
			'gamma': map(lambda x: 2 ** x, range(-10, -5)), 'epsilon': map(lambda x: 2 ** x, range(-5, 0))}
	]

	# param_grid = [
	# 	{'C': map(lambda x: 2 ** x, range(5, 10)), 'kernel': ['linear'],
	# 		'epsilon': map(lambda x: 2 ** x, range(-10, -5))}
	# ]

	best = GridSearchCV(estimator = SVR(), param_grid = param_grid, cv = 5)
	best.fit(x_train, y_train)
	opt_hyperparameters = best.best_params_
	print opt_hyperparameters

	regressor = SVR(**opt_hyperparameters)

	regressor.fit(x_train, y_train)
	
	result = regressor.predict(x_test)
	
	# train_error = (np.mean(map(lambda x, y: (x - y) ** 2, regressor.predict(x_train), y_train)))
	# test_error = (np.mean(map(lambda x, y: (x - y) ** 2, regressor.predict(x_test), y_test)))
	# print train_error, test_error
	# return train_error, test_error
	
	# if interested in accuracy
	train_score = regressor.score(x_train, y_train)
	test_score = regressor.score(x_test, y_test)
	print train_score, test_score
	return train_score, test_score

if __name__ == '__main__':

	random.seed(CONST_RANDOM_SEED)

	if(len(sys.argv) == 1):
		print "Usage:\nFirst Argument:\n0 for mat.csv\n1 for por.csv\nSecond Argument:\nl1 for LASSO\nl2 for Ridge"
		sys.exit(0)
	choice = sys.argv[1]
	if(choice == '1'):
		choice = '-por.csv'
	elif(choice == '0'):
		choice = '-mat.csv'
	else:
		print "Usage:\n0 for mat.csv\n1 for por.csv"
		sys.exit(0)		

	file_name = os.path.join(os.path.dirname(__file__), 'data/'+ 'transformed-student' + choice)
	penalty = sys.argv[2] 								# l1 for LASSO, l2 for ridge
	degree = 1											# atmost degree of polynomial features
	k = 10 												# for k-fold cross validation
	train_data_perc = 90
	max_iter = 5 										# number of iterations of SGD
	
	x_axis = [] 										# train_data_perc
	y_train, y_test = [], []

	for train_data_perc in range(30, 80, 10):
		train_error, test_error = svm(file_name, train_data_perc, True, True)
		y_train.append(train_error)
		y_test.append(test_error)
		x_axis.append(train_data_perc)

	plt.plot(x_axis, y_train, label = 'train')
	plt.plot(x_axis, y_test, label = 'test')
	plt.legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3, ncol = 2, mode = "expand", borderaxespad = 0.)
	plt.show()