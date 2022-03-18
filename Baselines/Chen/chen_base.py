import time

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import numpy as np
import pandas as pd

from Data.simulation_settings import *

#from sklearn.preprocessing import StandardScaler

#from Baselines.indirect.grid_fct import *

r = ro.r # object to execute and convert R code in python code
r['source']('Baselines/Chen/train_fn.r') # R-script that we execute from python

# Loading the function we defined in R-script
train_fn = ro.globalenv['naive_chen_train']
test_fn = ro.globalenv['naive_chen_test']

def fit_chen(args):
	"""
	function to fit Chen-Baseline-Method
	"""
	dt = which_setting(args) # get data

	# Training-Validation-Testing-Split
	X_train, X_test, t_obs_train, t_obs_test, y_train, y_test, t_opt_train, t_opt_test, t_dens_train, t_dens_test= train_test_split(dt.x, dt.t_obs, dt.y_obs, dt.t_opt, dt.t_dens_obs, test_size=dt.n_test/dt.n, random_state=dt.seed)
	
	dt_train = pd.DataFrame(data={
			't_obs_train':		t_obs_train.flatten(),
			't_opt_train':		t_opt_train.flatten(),
			'y_obs_train':		y_train.flatten(),
			't_dens_obs_train':	t_dens_train.flatten()})

	dt_train =pd.concat([dt_train, pd.DataFrame(X_train)], axis=1)
	
	dt_test = pd.DataFrame(data={
			't_obs_test':		t_obs_test.flatten(),
			't_opt_test':		t_opt_test.flatten(),
			'y_obs_test':		y_test.flatten(),
			't_dens_obs_test':	t_dens_test.flatten()})
	dt_test =pd.concat([dt_test, pd.DataFrame(X_test)], axis=1)
	
	# convert pandas objects into r objects
	with localconverter(ro.default_converter + pandas2ri.converter):
		dt_train_r = ro.conversion.py2rpy(dt_train)
		dt_test_r = ro.conversion.py2rpy(dt_test)

	# Execute R-functions on r objects in python
	pred_train = train_fn(dt_train_r)
	pred_test = test_fn(dt_train_r, dt_test_r)

	# Convert R-objects back to python objects (the results of fitting)
	with localconverter(ro.default_converter + pandas2ri.converter):
		pred_train_pd = ro.conversion.rpy2py(pred_train)
		pred_test_pd = ro.conversion.rpy2py(pred_test)

	# Calculate the sum of the counterfactual outcomes for above IDR
	value_train=dt.generate_value(x=torch.Tensor(X_train)[:,:9].to(dt.device), t_obs=torch.Tensor(pred_train_pd.pred_train).unsqueeze(1).to(dt.device), t_opt=torch.Tensor(t_opt_train).to(dt.device))/(dt.n_train+dt.n_val)
	value_test=dt.generate_value(x=torch.Tensor(X_test)[:,:9].to(dt.device), t_obs=torch.Tensor(pred_test_pd.pred_test).unsqueeze(1).to(dt.device), t_opt=torch.Tensor(t_opt_test).to(dt.device))/dt.n_test

	return value_train, value_test

def fit_chen_repeat(args):
	"""
	funciton to fit Chen-Baseline-Method multiple times
	"""
	if args.verbose:
		start = time.time()
		print(f'Start treatment model fitting: size: {args.n_train+args.n_val}')
	
	res = np.array([fit_chen(args) for i in range(args.n_repeats)])
	
	if args.verbose:
		end = time.time()
		print(f"Training time: {(end-start)/60:.3f} minutes")
		print(f"Chen mean_test: {np.round(np.mean(res[:,1]), decimals=4):.3f}")

	return {'mean_test': np.round(np.mean(res[:,1]), decimals=4),
		'std_test': np.round(np.std(res[:,1]), decimals=2),
		'mean_train': np.round(np.mean(res[:,0]), decimals=4),
		'std_train':  np.round(np.std(res[:,0]), decimals=2)}
