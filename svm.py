from sklearn.svm import SVR
import linear_regression

CONST_RANDOM_SEED = 69

def svm(dataset_file_name, train_data_perc, shuffle = False, random_seed = CONST_RANDOM_SEED):
	x_train, y_train, x_test, y_test = get_train_test_data(dataset_file_name, train_data_perc, degree, shuffle)

	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	svr_lin = SVR(kernel='linear', C=1e3)
	svr_poly = SVR(kernel='poly', C=1e3, degree=2)
	# y_rbf = svr_rbf.fit(x_train, y_train).predict(x_test)
	# print np.mean(map(lambda x, y: (x - y) ** 2, y_rbf, y_test))
	y_lin = svr_lin.fit(x_train, y_train).predict(x_test)
	print np.mean(map(lambda x, y: (x - y) ** 2, y_lin, y_test))
	# y_poly = svr_poly.fit(x_train, y_train).predict(x_test)
	# print np.mean(map(lambda x, y: (x - y) ** 2, y_poly, y_test))