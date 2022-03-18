import numpy as np

def build_grid_t(X, grid_size=5, n_cov=3):
	"""
	function to build grid of potential treatment values, for simplicity we use the range of X
	"""
	grid_t = np.linspace(X[:,n_cov].min(),X[:,n_cov].max(), num=grid_size)
	grid_t.resize((grid_size,1))
	return grid_t

def build_grid(obs,grid_t):
	"""
	function: uses the grid of potential treatments and bindes that grid to the covariates of one observation
	"""
	grid_x = np.tile(obs,(len(grid_t),1)) # repeat X_i and match to every potential treatment
	return np.concatenate((grid_x,grid_t), axis=1)

def max_grid(clf, grid):
	"""
	function to find the treatment that resulted in the largest outcome for model, clf, from the grid of potential outcomes
	:params clf: classifier as Sklearn-class-object
	:params grid: grid with one observations repeated multiple times for a range of potential treatments
	"""
	pred=clf.predict(grid)
	pred_max=np.argmax(pred, axis=0)
	return grid[pred_max,-1]

def inverse_IDR(clf, X, grid_size=50, n_cov=3):
	"""
	function to bind all above functions together and find the optimal treatments for all observations
	:params clf: classifier as Sklearn-class-object
	:params X: covariate matrix
	:params grid_size: number of potential treatments per observations
	"""
	grid_t=build_grid_t(X=X, grid_size=grid_size, n_cov=n_cov)
	dose_max=np.array([]) # array with optimal doses/treatments
	
	for i in range(X.shape[0]):
		grid=build_grid(X[i,:n_cov], grid_t)
		dose_max=np.append(dose_max, max_grid(clf,grid))
	return dose_max