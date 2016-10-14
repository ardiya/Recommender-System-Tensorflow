import pandas as pd
import numpy as np
from numpy.random import random
from sklearn import cross_validation

def _load(file, sep):
	udata = pd.read_csv(file, sep=sep, names=['userid', 'itemid', 'rating', 'timestamp'], engine='python')
	udata['rating'] = (udata['rating'] - 3) / 2

	map_uid_index = {uid:idx for idx, uid in enumerate(udata['userid'].unique())}
	map_iid_index = {iid:idx for idx, iid in enumerate(udata['itemid'].unique())}

	total_user = len(map_uid_index.items())
	total_item = len(map_iid_index.items())
	print("Total users:", total_user)
	print("Total items:", total_item)

	train_matrix = np.zeros([total_user, total_item], dtype=np.float32)
	test_matrix = np.zeros([total_user, total_item], dtype=np.float32)

	for line in udata.itertuples():
		if random() <0.9:
			train_matrix[map_uid_index[line[1]],
				map_iid_index[line[2]]] = line[3]
		else:
			test_matrix[map_uid_index[line[1]],
				map_iid_index[line[2]]] = line[3]

	return train_matrix, test_matrix

def load_movielens_100k():
	return _load('ml-100k/u.data', '\t')

def load_movielens_1m():
	return _load('ml-1m/ratings.dat', '::')