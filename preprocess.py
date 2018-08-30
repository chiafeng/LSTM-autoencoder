import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, StandardScaler

def get_batch(filedir, batch_size, time_steps, step_dims):
	dataset = pd.read_csv(filedir, header=None, dtype=np.float32)
	X = dataset.values
	X[np.isnan(X)] = 0

	# discard timestamp
	X = X[:, 1:]

	#TODO: check X.shape is match step_dims

	# standardization
	scaler = StandardScaler()
	X = scaler.fit_transform(X)		# StandardScaler expect dim <= 2

	# truncate last part to let X reshape into (?, time_steps, step_dims)
	remainder = X.shape[0] % time_steps
	X = X[:(X.shape[0]-remainder)]
	X = np.reshape(X, [-1, time_steps, step_dims])

	# randomly select batch_size samples from X
	np.random.seed(27)
	batch_indices = np.random.permutation(len(X))[:batch_size]
	X = X[batch_indices]
	batch_data = []
	for instance in X:
		batch_data.append(np.reshape(instance, [1, time_steps, step_dims]))

	# batch_data is list of array
	# len(batch_data) == batch_size
	# batch_data[i].shape == (1, time_steps, step_dims)

	return batch_data