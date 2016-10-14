import numpy as np
import random
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn.svm import SVR

import os
import sys

CONST_RANDOM_SEED = 69

def process(dataset_file_name, shuffle = False):
	
	x, y = [], []
	data = []

	with open(dataset_file_name, 'r') as f:
		for line in f:
			data.append(line)

	if shuffle:
		random.shuffle(data)

	for line in data:
		line = line.split(",")
		x.append(map(int, line[:-1]))
		y.append(int(line[-1:][0].split('\n')[0]))

	return x, y

def k_fold_cross_validation(random_seed, x, y, delta, k, penalty):
	MSE = 0.0
	regressor = linear_model.SGDRegressor(random_state = random_seed, alpha = delta, penalty = penalty)
	scores = cross_val_score(regressor, x, y, cv = k)
	# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	accuracy = scores.mean()
	return accuracy

	""" self made cross validator """
	# kf = KFold(n_splits = k)
	# scores = cross_val_score(clf, iris.data, iris.target, cv=5)
	# x, y = np.array(x), np.array(y)
	# for train, test in kf.split(x):
	# 	x_train, y_train = x[train], y[train]
	# 	x_test, y_test = x[test], y[test]
	# 	regressor.fit(x_train, y_train);
	# 	result = regressor.predict(x_test)
	# 	MSE += np.mean(map(lambda x, y: (x - y) ** 2, result, y_test))
	# return MSE / k

def find_optimal_ridge_parameter(random_seed, x, y, k, penalty):
	optimal_delta, optimal_accuracy = 0, 0
	candidates = [0] + map(lambda x: 0.01 * (2 ** x), range(11))
	x_axis, y_axis = [], []
	for delta in candidates:
		temp_accuracy = k_fold_cross_validation(random_seed, x, y, delta, k, penalty)
		if(temp_accuracy > optimal_accuracy):
			optimal_delta, optimal_accuracy = delta, temp_accuracy
		x_axis.append(delta)
		y_axis.append(temp_accuracy)
	
	# plt.plot(x_axis, y_axis)
	# plt.show()
	return optimal_delta

def svm(dataset_file_name, train_data_perc, shuffle = False, random_seed = CONST_RANDOM_SEED):
	x, y = process(dataset_file_name, shuffle)

	train_data_length = train_data_perc * len(x) / 100
	x_train, y_train = x[:train_data_length], y[:train_data_length]
	x_test, y_test = x[train_data_length:], y[train_data_length:]

	# to normalize the data attributes
	scaler = StandardScaler(); scaler.fit(x_train)
	
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)

	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	svr_lin = SVR(kernel='linear', C=1e3)
	svr_poly = SVR(kernel='poly', C=1e3, degree=2)
	# y_rbf = svr_rbf.fit(x_train, y_train).predict(x_test)
	# print np.mean(map(lambda x, y: (x - y) ** 2, y_rbf, y_test))
	y_lin = svr_lin.fit(x_train, y_train).predict(x_test)
	print np.mean(map(lambda x, y: (x - y) ** 2, y_lin, y_test))
	# y_poly = svr_poly.fit(x_train, y_train).predict(x_test)
	# print np.mean(map(lambda x, y: (x - y) ** 2, y_poly, y_test))

def linear_regression(dataset_file_name, train_data_perc, penalty, k, shuffle = False, random_seed = CONST_RANDOM_SEED):

	x, y = process(dataset_file_name, shuffle)

	train_data_length = train_data_perc * len(x) / 100
	x_train, y_train = x[:train_data_length], y[:train_data_length]
	x_test, y_test = x[train_data_length:], y[train_data_length:]

	# print train_data_perc, len(x_train)
	
	# to normalize the data attributes
	scaler = StandardScaler(); scaler.fit(x_train)
	
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)

	#SGD

	opt_delta = find_optimal_ridge_parameter(random_seed, x_train, y_train, k, penalty)
	# print opt_delta
	regressor = linear_model.SGDRegressor(random_state = random_seed, alpha = opt_delta, penalty = penalty)
	regressor.fit(x_train, y_train);

	# result = regressor.predict(x_test)

	# print "SGD solution:"
	# # print regressor.coef_
	# print "MSE: ", np.mean(map(lambda x, y: (x - y) ** 2, result, y_test))
	# print "Regressor score: ", regressor.score(x_test, y_test)

	# #Exact Least Squares solution

	# regressor = linear_model.LinearRegression()
	# regressor.fit(x_train, y_train); result = regressor.predict(x_test)

	# print "Least Squares solution:"
	# # print regressor.coef_
	# print "MSE: ", np.mean(map(lambda x, y: (x - y) ** 2, result, y_test))
	# print "Regressor score: ", regressor.score(x_test, y_test)

	# if interested in RMSE
	train_error = (np.mean(map(lambda x, y: (x - y) ** 2, regressor.predict(x_train), y_train)) ** 0.5)
	test_error = (np.mean(map(lambda x, y: (x - y) ** 2, regressor.predict(x_test), y_test)) ** 0.5)
	return train_error, test_error

	# # if interested in accuracy
	# train_score = regressor.score(x_train, y_train)
	# test_score = regressor.score(x_test, y_test)
	# return train_score, test_score

def plot_learning_curves(dataset_file_name, penalty, k, start = 10, stop = 100, step = 10, random_seed = CONST_RANDOM_SEED):
	x_axis = [] # train_data_perc
	y_train, y_test = [], [] # MSE
	for train_data_perc in range(start, stop, step):
		train_error, test_error = linear_regression(file_name, train_data_perc, penalty, k, True)
		x_axis.append(train_data_perc)
		y_train.append(train_error)
		y_test.append(test_error)

	plt.plot(x_axis, y_train, label = 'train')
	plt.plot(x_axis, y_test, label = 'test')
	plt.legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3, ncol = 2, mode = "expand", borderaxespad = 0.)
	plt.show()

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
	penalty = sys.argv[2] # l1 for LASSO, l2 for ridge
	
	# linear_regression(file_name, 50, penalty, 10, True)
	# svm(file_name, 50, True)
	
	plot_learning_curves(file_name, penalty, 10)