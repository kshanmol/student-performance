import random

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

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
		x.append(map(float, line[:-1]))
		y.append(float(line[-1:][0].split('\n')[0]))

	return x, y

def reduced_features(x, threshold = 0.99):
	pca = PCA()
	output_data = pca.fit_transform(x)

	#Find k which explains 95% of variance

	# print len(x[0])
	k = 0
	explained = 0
	while(explained < threshold):
		explained += pca.explained_variance_ratio_[k]
		k += 1
	pca = PCA(k)
	x = pca.fit_transform(x)
	
	# print len(x[0])
	return x