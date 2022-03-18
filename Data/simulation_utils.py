import numpy as np
import torch 

def power_x_fn(dt, coef, pw):
	"""
	function to clean code: raise x to the power of pw and matmul in numpy
	"""
	return np.expand_dims(np.matmul(np.power(dt, pw), coef), axis=1)

def power_x_fn_tensor(dt, coef, pw):
	"""
	function to clean code: raise x to the power of pw and matmul in torch tensors
	"""
	return torch.matmul(dt.pow(pw), coef).unsqueeze(1)

def power_t_fn(t, coef):
	"""
	function to clean code: raise t to the power of pw and matmul in numpy
	"""
	res = t*coef[0]
	for i in range(1,len(coef)):
		res = res + coef[i] * np.power(t, i+1)
	return res

def power_t_fn_tensor(t, coef):
	"""
	function to clean code: raise t to the power of pw and matmul in torch tensors
	"""
	return torch.cat([coef[i]*t**(i+1) for i in range(len(coef))], dim=1).sum(dim=1, keepdim=True)
	
def interaction_fn(x,t,coef):
	"""
	function to clean code: interaction effeect between x and t in numpy
	"""
	res = np.zeros(t.shape)
	for i in range(len(coef)):
		res = res + np.expand_dims(x[:,i]*coef[i], axis=1) * t
	return res

def alpha_x_y(n_cov):
	"""
	function to clean code: baseline effect from x to y in numpy
	"""
	params = [0.5, -0.5]*30
	return params[:n_cov]

def alpha_x_y_tensor(n_cov):
	"""
	function to clean code: baseline effect from x to y in torch tensors
	"""
	params = [0.5, -0.5]*30
	return torch.Tensor(params[:n_cov])

def alpha_x_t(n_cov):
	"""
	function to clean code: baseline effect from x to t in numpy
	"""
	params = [1,.5,0.1]*30
	return params[:n_cov]

def alpha_x_t_tensor(n_cov):
	"""
	function to clean code: baseline effect from x to t in torch tensors
	"""
	params = [1,.5,0.1]*30
	return torch.Tensor(params[:n_cov])

def beta_t_y(pw):
	"""
	function to clean code: baseline effect from t to y in numpy
	"""
	params = [1.5, -0.3, 0, 0, 0]
	return params[:pw]

def beta_t_y_tensor(pw):
	"""
	function to clean code: baseline effect from t to y in tensors
	"""
	params = [1.5, -0.3, 0, 0, 0]
	return torch.Tensor(params[:pw])