import numpy as np
import random
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
import os
import sys

def find_optimal_ridge_parameter(random_state, low, high, x_test, y_test, x_train, y_train):
	for i in range(25):
		temp = [(2*low + high) / 3, (low + 2*high) / 3]
		regressor = linear_model.SGDRegressor(random_state=14, alpha=temp[0])
		regressor.fit(x_train, y_train); result = regressor.predict(x_test)
		val1 = np.mean(map(lambda x, y: (x - y) ** 2, result, y_test))

		regressor = linear_model.SGDRegressor(random_state=14, alpha=temp[1])
		regressor.fit(x_train, y_train); result = regressor.predict(x_test)
		val2 = np.mean(map(lambda x, y: (x - y) ** 2, result, y_test))

		if(val1 > val2): low = temp[0]
		else: high = temp[1]
	return (low + high) / 2

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


def linear_regression(dataset_file_name, train_data_perc):

	x,y = process(dataset_file_name, True)

	train_data_length = train_data_perc * len(x) / 100
	x_train, y_train = x[:train_data_length], y[:train_data_length]
	x_test, y_test = x[train_data_length:], y[train_data_length:]

	# to normalize the data attributes
	scaler = StandardScaler(); scaler.fit(x_train)
	
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)

	#SGD
	random_seed = 30
	min_alpha, max_alpha = 0.0, 100.0

	opt_alpha = find_optimal_ridge_parameter(random_seed, min_alpha, max_alpha, x_test, y_test, x_train, y_train)
	print opt_alpha
	regressor = linear_model.SGDRegressor(random_state=14, alpha=opt_alpha)
	regressor.fit(x_train, y_train); result = regressor.predict(x_test)

	print "SGD solution:"
	print regressor.coef_
	print "MSE: ", np.mean(map(lambda x, y: (x - y) ** 2, result, y_test))
	print "Regressor score: ", regressor.score(x_test, y_test)

	#Exact Least Squares solution

	regressor = linear_model.LinearRegression()
	regressor.fit(x_train, y_train); result = regressor.predict(x_test)

	print "Least Squares solution:"
	print regressor.coef_
	print "MSE: ", np.mean(map(lambda x, y: (x - y) ** 2, result, y_test))
	print "Regressor score: ", regressor.score(x_test, y_test)


if __name__ == '__main__':

	if(len(sys.argv) == 1):
		print "Usage:\n0 for mat.csv\n1 for por.csv"
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

	linear_regression(file_name, 95)