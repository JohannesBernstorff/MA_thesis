import numpy as np
import time

from sklearn.model_selection import GridSearchCV

import xgboost as xgb

from Baselines.indirect.grid_fct import *
from Data.simulation_settings import *

def fit_xgb(args, i):
	"""
	function to fit cross-validated gradient boosted tree regression (XGBoost)
	"""
	#args.seed = args.seed * i
	dt = which_setting(args)
	
	dt.t_obs.resize((len(dt.t_obs),1))
	X = np.concatenate((dt.x,dt.t_obs), axis=1)

	X_train, X_test, y_train, y_test, t_opt_train, t_opt_test, _, _= train_test_split(X, dt.y_obs, dt.t_opt, dt.t_dens_obs, test_size=dt.n_test/dt.n, random_state=dt.seed)
	#X_train, X_val, y_train, y_val, t_opt_train, t_opt_val = train_test_split(X_train, y_train, t_opt_train, test_size=dt.n_val/(dt.n_train+dt.n_val), random_state=dt.seed*2)
	
	xgb_model=xgb.XGBRegressor(subsample=0.7)
	
	parameters = {'max_depth': [2,4], 'n_estimators': [50,100, 150]}
	clf = GridSearchCV(xgb_model,parameters, verbose=1, n_jobs=1)
	clf.fit(X_train, y_train)
	
	# functions from grid_fct.py to evaluate indirect models
	t_pred = inverse_IDR(clf, X_train, grid_size=100, n_cov=args.n_cov)
	value_train = dt.generate_value(x=torch.Tensor(X_train)[:,:args.n_cov].to(dt.device), t_obs=torch.Tensor(t_pred).unsqueeze(1).to(dt.device), t_opt=torch.Tensor(t_opt_train).to(dt.device))/(dt.n_train+dt.n_val)

	t_pred = inverse_IDR(clf, X_test, grid_size=100, n_cov=args.n_cov)
	value_test=dt.generate_value(x=torch.Tensor(X_test)[:,:args.n_cov].to(dt.device), t_obs=torch.Tensor(t_pred).to(dt.device).unsqueeze(1), t_opt=torch.Tensor(t_opt_test).to(dt.device))/dt.n_test

	return value_train, value_test

def fit_xgb_repeat(args):
	"""
	function to fit cross-validated gradient boosted tree regression (XGBoost) multiple times
	"""
	if args.verbose:
		start = time.time()
		print(f'XGB: Start treatment model fitting: size: {args.n_train+args.n_val}')
	
	res = np.array([fit_xgb(args, i) for i in range(args.n_repeats)])
	
	if args.verbose:
		end = time.time()
		print(f"Training time: {(end-start)/60:.3f} minutes")

	return {'mean_test': np.round(np.mean(res[:,1]), decimals=4),
		'std_test': np.round(np.std(res[:,1]), decimals=2),
		'mean_train': np.round(np.mean(res[:,0]), decimals=4),
		'std_train':  np.round(np.std(res[:,0]), decimals=2)}
