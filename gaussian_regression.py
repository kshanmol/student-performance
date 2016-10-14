import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor

import os
import sys

CONST_RANDOM_SEED = 69

# def generate_centres(x, num_centres, random_seed = CONST_RANDOM_SEED):
# 	if(len(x) == 0):
# 		print "No Data"
# 		return
# 	ranges = []
# 	for i in range(len(x[0])):
# 		min_val, max_val = x[0][i], x[0][i]
# 		for j in range(len(x)):
# 			min_val = min(min_val, x[j][i])
# 			max_val = min(max_val, x[j][i])
# 		ranges.append([min_val, max_val])

# 	for i in range(k):
# 		centres.append(map(lambda x: random.uniform(x[0], x[1]), ranges))

# 	return centres

# def gaussian_basis(centres, width = 1):

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

def gaussian_regression(dataset_file_name, train_data_perc, shuffle = False, random_seed = CONST_RANDOM_SEED):

	x, y = process(dataset_file_name, shuffle)

	train_data_length = train_data_perc * len(x) / 100
	x_train, y_train = x[:train_data_length], y[:train_data_length]
	x_test, y_test = x[train_data_length:], y[train_data_length:]
	
	# to normalize the data attributes
	scaler = StandardScaler(); scaler.fit(x_train)
	
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)

	# opt_delta = find_optimal_ridge_parameter(random_seed, x_train, y_train, k, penalty)
	# # print opt_delta

	regressor = GaussianProcessRegressor(random_state = random_seed, n_restarts_optimizer = 100000000, alpha = 1e-6)
	regressor.fit(x_train, y_train);

	# if interested in accuracy
	train_score = regressor.score(x_train, y_train)
	test_score = regressor.score(x_test, y_test)

	print test_score

	return train_score, test_score

if __name__ == '__main__':

	random.seed(CONST_RANDOM_SEED)

	if(len(sys.argv) == 1):
		print "Usage:\nFirst Argument:\n0 for mat.csv\n1 for por.csv"
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
	
	gaussian_regression(file_name, 60, True)
	
	# plot_learning_curves(file_name, penalty, 10)