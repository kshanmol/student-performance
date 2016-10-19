import os
import sys
import random

from sklearn import datasets, linear_model
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import get_data

import numpy as np
import matplotlib.pyplot as plt

CONST_RANDOM_SEED = 20

def get_train_test_data(dataset_file_name, train_data_perc, degree, shuffle = True, do_PCA = False, PCA_threshold = 0.99):
	x, y = get_data.process(dataset_file_name, shuffle)

	poly = PolynomialFeatures(degree = degree, interaction_only = False, include_bias = False)
	x = poly.fit_transform(x)

	train_data_length = train_data_perc * len(x) / 100
	x_train, y_train = x[:train_data_length], y[:train_data_length]
	x_test, y_test = x[train_data_length:], y[train_data_length:]
	
	if(do_PCA):
		x_train, x_test = get_data.reduced_features(x_train, x_test, PCA_threshold)

	# to normalize the data attributes
	scaler = StandardScaler(); scaler.fit(x_train)
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)

	return x_train, y_train, x_test, y_test

def exact_least_squares(dataset_file_name, train_data_perc, degree, shuffle = True):
	x_train, y_train, x_test, y_test = get_train_test_data(dataset_file_name, train_data_perc, degree, shuffle)

	regressor = linear_model.LinearRegression()
	regressor.fit(x_train, y_train);

	result = regressor.predict(x_test)

	print "Least Squares solution:"
	# print regressor.coef_
	print "MSE: ", np.mean(map(lambda x, y: (x - y) ** 2, result, y_test))
	print "Regressor score: ", regressor.score(x_test, y_test)

def linear_regression(dataset_file_name, train_data_perc, penalty, k, degree, max_iter = 5, shuffle = True, do_PCA = False, random_seed = CONST_RANDOM_SEED):
	random.seed(random_seed)

	x_train, y_train, x_test, y_test = get_train_test_data(dataset_file_name, train_data_perc, degree, shuffle, do_PCA)

	#SGD

	regressor = linear_model.SGDRegressor()

	# param_grid = [
	# 	{'alpha': map(lambda x: 2 ** x, range(-10, 10)), 'random_state': [random_seed], 'penalty': ['l1', 'l2'],
	# 		'n_iter': range(max_iter, max_iter + 1)}
	# ]

	# best = GridSearchCV(estimator = regressor, param_grid = param_grid, cv = 10)
	# best.fit(x_train, y_train)

	# opt_hyperparameters = best.best_params_
	opt_hyperparameters = {'penalty': 'l1', 'alpha': 0.0625, 'random_state': random_seed, 'n_iter': 20}
	# print opt_hyperparameters

	regressor = linear_model.SGDRegressor(**opt_hyperparameters)
	regressor.fit(x_train, y_train);

	train_error = (np.mean(map(lambda x, y: (x - y) ** 2, regressor.predict(x_train), y_train)) ** 0.5)
	test_error = (np.mean(map(lambda x, y: (x - y) ** 2, regressor.predict(x_test), y_test)) ** 0.5)
	# print train_data_perc, degree, test_error
	# print "degree :", degree, "data_perc :", train_data_perc, "test error :", test_error, "train error: ", train_error

	return train_error, test_error
	
	# # if interested in accuracy
	# train_score = regressor.score(x_train, y_train)
	# test_score = regressor.score(x_test, y_test)
	# return train_score, test_score

def plot_learning_curves(dataset_file_name, penalty, k, degree, max_iter = 5, start = 10, stop = 100, step = 10, shuffle = True, do_PCA = False, random_seed = CONST_RANDOM_SEED):
	x_axis = [] 										# train_data_perc
	y_train, y_test = {}, {}

	for train_data_perc in range(start, stop, step):
		for d in degree:
			if d not in y_train: y_train[d] = []
			if d not in y_test: y_test[d] = []
			train_error, test_error = linear_regression(file_name, train_data_perc, penalty, k, d, max_iter, shuffle, do_PCA, random_seed)
			print "degree :", d, "data_perc :", train_data_perc, "test error :", test_error, "train error: ", train_error
			y_train[d].append(train_error)
			y_test[d].append(test_error)
		x_axis.append(train_data_perc)

	plt.xlabel("Train Data Percentage")
	plt.ylabel("Root Mean Square Error")
	for d in degree:
		plt.plot(x_axis, y_test[d], label = 'test')
		# plt.plot(x_axis, y_train[d], label = 'train')
		print x_axis
		print y_test[d]
		# print y_train[d]

	
	if(dataset_file_name.find('mat') != -1): plt.title('Math Course')
	elif(dataset_file_name.find('por') != -1): plt.title('Portuguese Course')
	plt.legend(loc='upper right')
	# plt.legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3, ncol = 2, mode = "expand", borderaxespad = 0.)
	plt.show()

if __name__ == '__main__':
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
	degree = 2											# atmost degree of polynomial features
	k = 10 												# for k-fold cross validation
	train_data_perc = 80
	max_iter = 20 										# number of iterations of SGD
	
	# linear_regression(file_name, train_data_perc, penalty, k, degree, max_iter, do_PCA = True)
	
	plot_learning_curves(file_name, penalty, k, [1, 2], max_iter, do_PCA = True)