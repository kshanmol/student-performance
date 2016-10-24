import os
import sys
import random

from sklearn import datasets, linear_model
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

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

def random_forest_regression(dataset_file_name, train_data_perc, k, degree, shuffle = True, do_PCA = False, random_seed = CONST_RANDOM_SEED):
	
	random.seed(random_seed)

	x_train, y_train, x_test, y_test = get_train_test_data(dataset_file_name, train_data_perc, degree, shuffle, do_PCA)

	regressor = RandomForestRegressor(random_state=random_seed)
	param_grid = [
		{'n_estimators': [45]
		,'random_state': [random_seed]
		,'max_features': ['auto']
		#,'min_samples_split' : [2,3,4,5,6,7]
		}
	]
	
	# best = GridSearchCV(estimator = regressor, param_grid = param_grid, cv = 5)
	# best.fit(x_train, y_train)

	# means = best.cv_results_['mean_test_score']
	# stds = best.cv_results_['std_test_score']
	# for mean, std, params in zip(means, stds, best.cv_results_['params']):
	# 	print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

	# opt_hyperparameters = best.best_params_
	# print opt_hyperparameters
	opt_hyperparameters = {'n_estimators': 45
		,'random_state': random_seed
	}
	regressor = RandomForestRegressor(**opt_hyperparameters)
	regressor.fit(x_train, y_train);

	train_error = (np.mean(map(lambda x, y: (x - y) ** 2, regressor.predict(x_train), y_train)) ** 0.5)
	test_error = (np.mean(map(lambda x, y: (x - y) ** 2, regressor.predict(x_test), y_test)) ** 0.5)
	# print train_data_perc, degree, test_error
	# print "degree :", degree, "data_perc :", train_data_perc, "test error :", test_error, "train error: ", train_error

	return train_error, test_error

def plot_learning_curves(dataset_file_name, k, degree, start = 10, stop = 100, step = 10, shuffle = True, do_PCA = False, random_seed = CONST_RANDOM_SEED):
	x_axis = [] 										# train_data_perc
	y_train, y_test = {}, {}

	for train_data_perc in range(start, stop, step):
		for d in degree:
			if d not in y_train: y_train[d] = []
			if d not in y_test: y_test[d] = []
			train_error, test_error = random_forest_regression(file_name, train_data_perc, k, d, shuffle, do_PCA, random_seed)
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
	degree = 2											# atmost degree of polynomial features
	k = 10 												# for k-fold cross validation
	train_data_perc = 80
	
	#print random_forest_regression(file_name, train_data_perc, k, 1, do_PCA = True)
	
	plot_learning_curves(file_name, k, [1], do_PCA = True)