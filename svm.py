from sklearn.svm import SVR
import get_data
import random
import sys
import os
from sklearn.model_selection import GridSearchCV

import numpy as np

CONST_RANDOM_SEED = 69

def svm(dataset_file_name, train_data_perc, shuffle = True, do_PCA = False, PCA_threshold = 0.99, random_seed = CONST_RANDOM_SEED):
	x_train, y_train, x_test, y_test = get_data.get_train_test_data(dataset_file_name, train_data_perc, 1, shuffle, do_PCA, PCA_threshold)

	# svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1)

	param_grid = [
		{'C': map(lambda x: 2 ** x, range(-5, 10)), 'kernel': ['rbf']}
	]

	best = GridSearchCV(estimator = SVR(), param_grid = param_grid, cv = 5)
	best.fit(x_train, y_train)
	opt_hyperparameters = best.best_params_
	print opt_hyperparameters

	svr_lin = SVR(**opt_hyperparameters)
	# svr_poly = SVR(kernel = 'poly', C = 1e3, degree = 2)
	# y_rbf = svr_rbf.fit(x_train, y_train).predict(x_test)
	# print np.mean(map(lambda x, y: (x - y) ** 2, y_rbf, y_test))
	y_lin = svr_lin.fit(x_train, y_train).predict(x_test)
	print np.mean(map(lambda x, y: (x - y) ** 2, y_lin, y_test))
	# y_poly = svr_poly.fit(x_train, y_train).predict(x_test)
	# print np.mean(map(lambda x, y: (x - y) ** 2, y_poly, y_test))

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
	
	# linear_regression(file_name, train_data_perc, penalty, k, degree, max_iter)
	for train_data_perc in range(10, 100, 10):
		svm(file_name, train_data_perc, True, True, 0.95)