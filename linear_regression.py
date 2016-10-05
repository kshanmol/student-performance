import numpy as np
import random
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

import os
import sys

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

def find_optimal_ridge_parameter(random_seed, low, high, x, y, k, penalty):
	for i in range(25):
		temp = [(2*low + high) / 3, (low + 2*high) / 3]
		val = []
		for j in range(2):
			val.append(k_fold_cross_validation(random_seed, x, y, temp[j], k, penalty))

		if(val[0] < val[1]): low = temp[0]
		else: high = temp[1]
	return (low + high) / 2

def linear_regression(dataset_file_name, train_data_perc, penalty, shuffle = False, random_seed = 14):

	x, y = process(dataset_file_name, shuffle)

	train_data_length = train_data_perc * len(x) / 100
	x_train, y_train = x[:train_data_length], y[:train_data_length]
	x_test, y_test = x[train_data_length:], y[train_data_length:]

	# to normalize the data attributes
	scaler = StandardScaler(); scaler.fit(x_train)
	
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)

	#SGD
	min_delta, max_delta = 0.0, 100.0

	opt_delta = find_optimal_ridge_parameter(random_seed, min_delta, max_delta, x_train, y_train, 10, penalty)
	print opt_delta
	regressor = linear_model.SGDRegressor(random_state = random_seed, alpha = opt_delta, penalty = penalty)
	regressor.fit(x_train, y_train);

	result = regressor.predict(x_test)

	print "SGD solution:"
	# print regressor.coef_
	print "MSE: ", np.mean(map(lambda x, y: (x - y) ** 2, result, y_test))
	print "Regressor score: ", regressor.score(x_test, y_test)

	#Exact Least Squares solution

	regressor = linear_model.LinearRegression()
	regressor.fit(x_train, y_train); result = regressor.predict(x_test)

	print "Least Squares solution:"
	# print regressor.coef_
	print "MSE: ", np.mean(map(lambda x, y: (x - y) ** 2, result, y_test))
	print "Regressor score: ", regressor.score(x_test, y_test)


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
	penalty = sys.argv[2] # l1 for LASSO, l2 for ridge
	linear_regression(file_name, 50, penalty, True)