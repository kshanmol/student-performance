import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV

import get_data

import os
import sys

CONST_RANDOM_SEED = 69

def gaussian_regression(dataset_file_name, train_data_perc, shuffle = False, random_seed = CONST_RANDOM_SEED):

	x, y = get_data.process(dataset_file_name, shuffle)
	
	x = get_data.reduced_features(x)

	train_data_length = train_data_perc * len(x) / 100
	x_train, y_train = x[:train_data_length], y[:train_data_length]
	x_test, y_test = x[train_data_length:], y[train_data_length:]
	
	# to normalize the data attributes
	scaler = StandardScaler(); scaler.fit(x_train)
	
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)

	regressor = GaussianProcessRegressor()

	param_grid = [
		{'alpha': map(lambda x: 2 ** x, range(-10, 10)), 'random_state': [CONST_RANDOM_SEED], 'n_restarts_optimizer': [100]}
	]

	best = GridSearchCV(estimator = regressor, param_grid = param_grid, cv = 5)
	best.fit(x_train, y_train)
	opt_hyperparameters = best.best_params_
	print opt_hyperparameters
	regressor = GaussianProcessRegressor(**opt_hyperparameters)
	regressor.fit(x_train, y_train)

	print regressor.kernel_
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
	
	for train_data_perc in range(10, 100, 10):
		gaussian_regression(file_name, train_data_perc, True)
	
	# plot_learning_curves(file_name, penalty, 10)







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