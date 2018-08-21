import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, StandardScaler

def get_batch(filedir, time_steps, step_dims, offset=0):
	dataset = pd.read_csv(filedir, header=None, dtype=np.float32)
	X = dataset.values
	X[np.isnan(X)] = 0

	# discard timestamp
	X = X[:, 1:]

	# discard useless part
	X = X[offset:time_steps]

	# standardization
	scaler = StandardScaler()
	X = scaler.fit_transform(X)		# StandardScaler expect dim <= 2

	X = np.reshape(X, [1, time_steps, step_dims])	# [samples, time_steps, step_dims]
	print(X.shape)
	# X = X.transpose(1, 0, 2)		# exchange 0th and 1st dimension for RNN input
	# print(X.shape)

	return X