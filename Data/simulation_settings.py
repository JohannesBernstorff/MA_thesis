import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.stats import norm, chi2, binned_statistic
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader, Subset

from Data.cde_own import *
from Data.DeepCDE.deepcde.bases.cosine import CosineBasis
from Data.DeepCDE.deepcde.deepcde_pytorch import cde_layer, cde_loss, cde_predict
from Data.DeepCDE.deepcde.utils import box_transform

from Data.simulation_utils import *


class own_sim_base():
	"""
	Base class for all simulation settings, used to define methods that are necessary for all subclasses, eg. build_dataloaders...
	"""
	def __init__(self, args):
		"""
		:params args: dict of arguments from args.parser
		"""
		# set all the class attributes. At some point we need these values and we do not want to pass args down in all operations 
		self.n_train, self.n_val, self.n_test = args.n_train, args.n_val, args.n_test
		self.n, self.n_cov, self.bins = self.n_train+self.n_val+self.n_test, args.n_cov, args.bins
		self.simulate_new, self.seed, self.scale_dt= args.simulate_new, args.seed, args.scale_dt
		self.device = 'cpu'
		self.noise, self.alpha_factor, self.delta_factor, self.gamma_factor = args.noise, args.alpha_factor, args.delta_factor, args.gamma_factor
		self.beta_1, self.beta_2 = args.beta_1, args.beta_2
		self.calculate_con_dens, self.with_true_t_dens = args.calculate_con_dens, args.with_true_t_dens
		
		if self.simulate_new: 
			self.simulate_data() # this operation is called after the subclass methods were defined
		else:
			self.load_data()
		self.train_test_split() 
		self.calculate_performance_metrics()
		self.set_device(num_gpu = args.num_gpu)

	def x_fn(self):
		"""
		placeholder function to be overwritten by setting subclass function
		"""
		pass
	
	def t_obs_fn(self):
		"""
		placeholder function to be overwritten by setting subclass function
		"""
		pass
	
	def t_opt_fn(self):
		"""
		placeholder function to be overwritten by setting subclass function
		"""
		pass
	
	def y_obs_fn(self):
		"""
		placeholder function to be overwritten by setting subclass function
		"""
		pass

	def t_dens_obs_fn(self):
		"""
		placeholder function to be overwritten by setting subclass function
		"""
		pass
	
	def generate_value(self, x, t_obs, t_opt):
		"""
		placeholder function to be overwritten by setting subclass function
		"""
		pass

	def load_data(self):
		"""
		function to load data from data_path
		"""
		dt=pd.read_csv(filepath_or_buffer=self.data_path).iloc[:self.n,:(self.n_cov+5)]
		self.t_obs = np.expand_dims(dt.t_obs.values, axis=1)
		self.y_obs = np.expand_dims(dt.y_obs.values, axis=1)
		self.t_opt = np.expand_dims(dt.t_opt.values, axis=1)
		self.t_dens_obs = np.expand_dims(dt.t_dens_obs.values, axis=1)
		x_i = [str(x) for x in range(self.n_cov)]
		self.x = dt.loc[:,x_i].values
		
	def simulate_data(self):
		"""
		method to generate new data. class the functions from the settings subclass. only call simulate_data() once below functions were overwritten
		"""
		self.x_fn()
		self.t_obs_fn()
		self.t_opt_fn()
		self.y_obs_fn()
		self.t_dens_obs_fn()

	def set_device(self, num_gpu):
		"""
		method for setting CUDA and GPU
		:params num_gpu: number of GPUs available, code does support multi-GPU training
		"""
		self.device = "cpu"
		if torch.cuda.is_available() and num_gpu > 0:
			self.device = "cuda"

	def train_test_split(self):		
		"""
		method to split data into trianing and testing data.
		returns nothing but saves datasets and key information (optimal outcome) as attribute to the class
		"""
		test_ind = np.random.choice(self.n, size=self.n_test, replace=False) # sample indices
		train_ind = np.array([x for x in range(self.n) if x not in test_ind]) # sample indices

		self.x_train, self.x_test = self.x[train_ind,:], self.x[test_ind,:]
		self.t_obs_train, self.t_obs_test = self.t_obs[train_ind], self.t_obs[test_ind]
		self.t_opt_train, self.t_opt_test = self.t_opt[train_ind], self.t_opt[test_ind]
		self.y_obs_train, self.y_obs_test = self.y_obs[train_ind], self.y_obs[test_ind]
		
		if self.calculate_con_dens: # if true, then the cond. dens. is estimated... We call run_cde(), that is DeepCDE
			self.t_dens_obs_train, self.t_dens_obs_test=run_cde(self.x_train, self.t_obs_train), self.t_dens_obs[test_ind]
			self.t_dens_obs_train = self.t_dens_obs_train/np.mean(self.t_dens_obs_train) + .05
		else:
			self.t_dens_obs_train, self.t_dens_obs_test = self.t_dens_obs[train_ind], self.t_dens_obs[test_ind]
		
		# Instantiate TorchCategorizer, to bin the continuous treatment variable
		self.categorizer=TorchCategorizer(bins=self.bins) 
		self.t_obs_bin_train, _=self.categorizer.fit_transform(self.t_obs_train)
		self.t_obs_bin_test, _=self.categorizer.transform(self.t_obs_test)
		
		# calculate the optimal training/testing value, only makes sense if t_opt is well-defined
		self.y_opt_mean_train = self.generate_value(torch.Tensor(self.x_train), torch.Tensor(self.t_opt_train), torch.Tensor(self.t_opt_train))/(self.n_train+self.n_val)
		self.y_opt_mean_test = self.generate_value(torch.Tensor(self.x_test), torch.Tensor(self.t_opt_test), torch.Tensor(self.t_opt_test))/self.n_test
		
	def bootstrap_split(self):
		"""
		method to bootstrap from training data. 
		"""
		ind = np.random.choice(self.n_val+self.n_train, size=self.n_val+self.n_train, replace=True) # WITH REPLACEMENT!
		oob = np.array([x for x in range(self.n_val+self.n_train) if x not in ind]) # use for validation
		
		self.n_train_boot, self.n_val_boot = len(ind), len(oob) # As we sample with replacement the number of bootstrapped validation points not n_val anymore
		return ind, oob

	def build_dataloader_out_mi(self, batch_size = 64):
		"""
		method to build dataloader for the outcome prediction in the Mi-Baseline. Only need X and Y nothing else in the first part of Mi.
		:params batch_size: size of batch for the training of the Mi-Baseline
		"""
		ind, oob = self.bootstrap_split() # only the indices

		x_train, x_val=torch.Tensor(self.x_train[ind,:]), torch.Tensor(self.x_train[oob,:])
		y_obs_train, y_obs_val=torch.Tensor(self.y_obs_train[ind]), torch.Tensor(self.y_obs_train[oob])
		
		if self.scale_dt: # Must scale now, as we cannot use validation data in TorchScaler
			self.scaler=TorchScaler(x_train) # sets mean and std
			x_train=self.scaler.transform(x_train) # scales data
			x_val=self.scaler.transform(x_val) # scales x_val based on x_train characteristics
		
		train_dataset=TensorDataset(x_train, y_obs_train)
		val_dataset=TensorDataset(x_val, y_obs_val)
				
		self.batch_size = batch_size
		train_dataloader=DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=False)
		val_dataloader=DataLoader(val_dataset, batch_size=self.n_val, shuffle=False, pin_memory=False)

		return train_dataloader, val_dataloader

	def build_dataloader_ITR_mi(self, pred_train, pred_test, batch_size):
		"""
		method to build dataloader for the treatment prediction (second part) of the Mi-Baseline. We transform the predictions and use one-hot-encoded treatments
		Again include ensemble learning capabilities
		:params pred_train: outcome predictions for the training data
		:params batch_size: size of batch for the training of the Mi-Baseline
		"""
		ind, oob = self.bootstrap_split()
		
		t_obs_bin_train, t_obs_bin_val = self.t_obs_bin_train[ind], self.t_obs_bin_train[oob]

		# calculate residuals from the outcome predictions and actual outcomes
		res_train, res_val = pred_train[ind].to('cpu')-torch.Tensor(self.y_obs_train[ind]), pred_train[oob].to('cpu')-torch.Tensor(self.y_obs_train[oob])
		
		# encode binned treatments in one-hot if residual positive else encode in one-cold
		one_hot_train=torch.nn.functional.one_hot(t_obs_bin_train.to(torch.int64))[:,0,:]
		t_train=torch.where(res_train<0,one_hot_train, 1-one_hot_train)
		
		one_hot_val=torch.nn.functional.one_hot(t_obs_bin_val.to(torch.int64))[:,0,:]
		t_val=torch.where(res_val<0,one_hot_val, 1-one_hot_val)
		
		x_train, x_val=torch.Tensor(self.x_train[ind,:]), torch.Tensor(self.x_train[oob,:])
		
		if self.scale_dt: # have to scale after training-validation-split
			self.scaler=TorchScaler(x_train)
			x_train=self.scaler.transform(x_train)
			x_val=self.scaler.transform(x_val) # scale based on x_train
		
		t_opt_train, t_opt_val = torch.Tensor(self.t_opt_train[ind]), torch.Tensor(self.t_opt_train[oob]) # no need to categorize
		
		train_dataset=TensorDataset(x_train, t_train.float(), t_opt_train, res_train.abs())
		val_dataset=TensorDataset(x_val, t_val.float(), t_opt_val, res_val.abs())
		
		train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
		val_dataloader=DataLoader(val_dataset, batch_size=self.n_val, shuffle=False, pin_memory=False)

		return train_dataloader, val_dataloader

	def build_dataloader_boot(self, batch_size=64):
		"""
		method to build the dataloader for bootstrapped estimates with our proposed methods
		:params batch_size: number of observations in each trainin batch
		"""
		ind, oob = self.bootstrap_split() # generate indices
		x_train, x_val=torch.Tensor(self.x_train[ind,:]), torch.Tensor(self.x_train[oob,:])
		
		if self.scale_dt: # have to scale after training-validation-split
			self.scaler=TorchScaler(x_train)
			x_train=self.scaler.transform(x_train)
			x_val=self.scaler.transform(x_val) # scale based on x_train

		# we need y_obs, t_obs, t_opt (for reference), t_dens (for confounder correction)
		y_obs_train, y_obs_val = torch.Tensor(self.y_obs_train[ind]), torch.Tensor(self.y_obs_train[oob])
		t_obs_train, t_obs_val = torch.Tensor(self.t_obs_train[ind]), torch.Tensor(self.t_obs_train[oob])
		t_opt_train, t_opt_val = torch.Tensor(self.t_opt_train[ind]), torch.Tensor(self.t_opt_train[oob])
		t_dens_obs_train, t_dens_obs_val = torch.Tensor(self.t_dens_obs_train[ind]), torch.Tensor(self.t_dens_obs_train[oob])
		
		train_dataset=TensorDataset(x_train, t_obs_train, t_opt_train, y_obs_train, t_dens_obs_train)
		val_dataset=TensorDataset(x_val, t_obs_val, t_opt_val, y_obs_val, t_dens_obs_val)
		
		train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
		val_dataloader=DataLoader(val_dataset, batch_size=self.n_val_boot, shuffle=False, pin_memory=False)

		return train_dataloader, val_dataloader

	def build_dataloader_test(self):
		"""
		method to build the dataloader with the training/testing data. We separate two dataloader methods because we only need testing data once while bootstrapping multiple times
		"""
		x_train, x_test=torch.Tensor(self.x_train), torch.Tensor(self.x_test)
		
		if self.scale_dt: # have to scale after training-validation-split
			self.scaler=TorchScaler(x_train)
			x_train=self.scaler.transform(x_train)
			x_test=self.scaler.transform(x_test) # scale based on x_train
		
		# we need y_obs, t_obs, t_opt (for reference), t_dens (for confounder correction)
		y_obs_train, y_obs_test = torch.Tensor(self.y_obs_train), torch.Tensor(self.y_obs_test)
		t_obs_train, t_obs_test = torch.Tensor(self.t_obs_train), torch.Tensor(self.t_obs_test)
		t_opt_train, t_opt_test = torch.Tensor(self.t_opt_train), torch.Tensor(self.t_opt_test)
		t_dens_obs_train, t_dens_obs_test = torch.Tensor(self.t_dens_obs_train), torch.Tensor(self.t_dens_obs_test)
		
		train_dataset=TensorDataset(x_train, t_obs_train, t_opt_train, y_obs_train, t_dens_obs_train)
		test_dataset=TensorDataset(x_test, t_obs_test, t_opt_test, y_obs_test, t_dens_obs_test)
		
		train_dataloader=DataLoader(train_dataset, batch_size=self.n_test, shuffle=False, pin_memory=False)
		test_dataloader=DataLoader(test_dataset, batch_size=self.n_test, shuffle=False, pin_memory=False)

		return train_dataloader, test_dataloader

	def return_X(self):
		"""
		method to return torch tensor with scaled X..
		"""
		x_train, x_test=torch.Tensor(self.x_train), torch.Tensor(self.x_test)
		if self.scale_dt:
			self.scaler=TorchScaler(x_train)
			x_train=self.scaler.transform(x_train)
			x_test=self.scaler.transform(x_test)
		return x_train, x_test

	def calculate_performance_metrics(self):
		"""
		method to calculate the performance measures for reporting
		"""
		self.y_obs_mean_train, self.y_obs_mean_test = self.y_obs_train.mean().item(), self.y_obs_test.mean().item()		
	
	def save_data(self):
		"""
		method to save data to data_path
		"""
		dt = pd.DataFrame(data={
			't_obs':	self.t_obs.flatten(),
			't_opt':	self.t_opt.flatten(),
			'y_obs':	self.y_obs.flatten(),
			't_dens_obs':	self.t_dens_obs.flatten()})
		dt =pd.concat([dt, pd.DataFrame(self.x)], axis=1)
		dt.to_csv(path_or_buf=self.data_path)

class setting1_1(own_sim_base):
	"""
	class for the simulation of Simple Setting 1.
	:params args: dict of arguments from args.parser()
	"""
	def __init__(self, args):
		self.data_path = args.path +f'Data/own_data/setting1_1/dt_{args.n_train+args.n_val}_{args.n_cov}.csv' # Where to save results?
		super().__init__(args) # initialize the methods from base class
	
	def x_fn(self):
		"""
		method to generate covariates
		"""
		self.x = multivariate_normal.rvs(mean=[0]*self.n_cov, cov=5, size=self.n)
		
	def t_obs_fn(self):
		"""
		method to generate treatments
		"""
		self.t_obs = np.expand_dims(chi2.rvs(df=2, size=self.n), axis=1)
	
	def t_opt_fn(self):
		"""
		method to generate optimal treatments
		"""
		self.t_opt = np.ones((self.n,1))*10 # higher treatments are better!
		
	def y_obs_fn(self):
		"""
		method to generate outcomes
		"""
		baseline_x = power_x_fn(dt=self.x, coef=alpha_x_y(self.n_cov), pw=1)
		baseline_t = power_t_fn(t=self.t_obs, coef=beta_t_y(pw=2))
		noise = np.expand_dims(norm.rvs(loc=0, scale = self.noise, size=self.n), axis=1)
		s = baseline_x + baseline_t + noise
		self.y_obs = 2/(1+np.exp(-s/np.std(s)))

	def t_dens_obs_fn(self):
		"""
		method to generate conditional densities
		"""
		self.t_dens_obs = np.ones((self.n,1)) # in this setting we do not need to calculate the conditional density, no need for confounder correction
	
	def generate_value(self, x, t_obs, t_opt):
		"""
		method to generate sum of counterfactual outcomes
		"""
		baseline_x = power_x_fn_tensor(dt=x, coef=alpha_x_y_tensor(self.n_cov).to(x.device), pw=1)
		baseline_t = power_t_fn_tensor(t=t_obs, coef=beta_t_y_tensor(pw=2)).to(t_obs.device)
		s = baseline_x + baseline_t
		return (2/(1+(-s/s.std()).exp())).sum().item()

class setting1_1a(own_sim_base):
	"""
	class for the simulation of Simple Setting 1 with interaction effects
	:params args: dict of arguments from args.parser()
	"""
	def __init__(self, args):
		self.data_path = args.path +f'Data/own_data/setting1_1a/dt_{args.n_train+args.n_val}_{args.n_cov}.csv'
		super().__init__(args)
	
	def x_fn(self):
		"""
		method to generate covariates
		"""
		self.x = multivariate_normal.rvs(mean=[0]*self.n_cov, cov=5, size=self.n)
		
	def t_obs_fn(self):
		"""
		method to generate treatments
		"""
		self.t_obs = np.expand_dims(chi2.rvs(df=2, size=self.n), axis=1)
	
	def t_opt_fn(self):
		"""
		method to generate optimal treatments
		"""
		self.t_opt = np.ones((self.n,1))*10
		
	def y_obs_fn(self):
		"""
		method to generate outcomes
		"""
		baseline_x = power_x_fn(dt=self.x, coef=alpha_x_y(self.n_cov), pw=1)
		baseline_t = power_t_fn(t=self.t_obs, coef=beta_t_y(pw=2))
		interaction = self.gamma_factor*self.t_obs * np.expand_dims(self.x[:,0], axis=1) + self.gamma_factor*self.t_obs * np.expand_dims(self.x[:,1], axis=1) + self.gamma_factor*self.t_obs * np.expand_dims(self.x[:,2], axis=1)
		nonlinear = self.delta_factor*np.expand_dims(np.where(self.x[:,0]<0, 1, 0) + self.delta_factor*np.where(self.x[:,1]<0, -1, 0) + self.delta_factor*np.where(self.x[:,2]<0,1, 0), axis=1)
		noise = np.expand_dims(norm.rvs(loc=0, scale = self.noise, size=self.n), axis=1)
		s = baseline_x + baseline_t + noise + interaction + nonlinear
		self.y_obs = 2/(1+np.exp(-s/np.std(s)))

	def t_dens_obs_fn(self):
		"""
		method to generate conditional densities
		"""
		self.t_dens_obs = np.ones((self.n,1))
	
	def generate_value(self, x, t_obs, t_opt):
		"""
		method to generate the sum of counterfactual outcomes
		"""
		baseline_x = power_x_fn_tensor(dt=x, coef=alpha_x_y_tensor(self.n_cov).to(x.device), pw=1)
		baseline_t = power_t_fn_tensor(t=t_obs, coef=beta_t_y_tensor(pw=2)).to(t_obs.device)
		interaction = self.gamma_factor*x[:,0].unsqueeze(1)*t_obs + self.gamma_factor*x[:,1].unsqueeze(1)*t_obs + self.gamma_factor*x[:,2].unsqueeze(1)*t_obs
		nonlinear = self.delta_factor*torch.where(x[:,0]<0, 1, 0).unsqueeze(1) -self.delta_factor*torch.where(x[:,1]<0, 1, 0).unsqueeze(1) + self.delta_factor*torch.where(x[:,2]<0, 1, 0).unsqueeze(1)
		noise = torch.normal(0, self.noise, size=baseline_t.size()).to(x.device)
		s = baseline_x + baseline_t + interaction + nonlinear
		return (2/(1+(-s/s.std()).exp())).sum().item()
			
class setting1_2(own_sim_base):
	"""
	class for the simulation of Simple Setting 2.
	:params args: dict of arguments from args.parser()
	"""
	def __init__(self, args):
		self.data_path = args.path +f'Data/own_data/setting1_2/dt_{args.n_train+args.n_val}_{args.n_cov}.csv'
		super().__init__(args)

	def x_fn(self):
		"""
		method to generate covariates
		"""
		self.x = multivariate_normal.rvs(mean=[0]*self.n_cov, cov=5, size=self.n)
		
	def t_obs_fn(self):
		"""
		method to generate treatments
		"""
		coef1 = np.array([1,0.5,0.1])
		baseline_x1 = power_x_fn(dt=self.x, coef=coef1, pw=1)
		
		coef2 = np.array([0.1,.05,0.01]) 
		baseline_x2 = np.expand_dims(np.matmul(np.power(self.x,2), coef2), axis=1)
		
		noise = np.expand_dims(norm.rvs(loc=0, scale = 0.25, size=self.n), axis=1)
		s = baseline_x1+ baseline_x2 + noise 
		self.t_obs = 5/(1+np.exp(-s/np.std(s)*2))
	
	def t_opt_fn(self):
		"""
		method to generate optimal treatments
		"""
		self.t_opt = np.ones((self.n,1))
		
	def y_obs_fn(self):
		"""
		method to generate outcomes
		"""
		power_t = 2*self.t_obs - 0.5*np.power(self.t_obs, 2)
		noise = np.expand_dims(norm.rvs(loc=0, scale = .25, size=self.n), axis=1)
		s = power_t+noise
		self.y_obs = 2/(1+np.exp(-s))
		
	def t_dens_obs_fn(self):
		"""
		method to generate conditional densities
		"""
		self.t_dens_obs = np.ones((self.n,1))
		
	def generate_value(self, x, t_obs, t_opt):
		"""
		method to generate the sum of counterfactual outcomes
		"""
		power_t = 2*t_obs - 0.5*t_obs.pow(2)
		value = 2/(1+(-power_t).exp())
		return value.sum().item()

class setting1_3(own_sim_base):
	"""
	class for the simulation of Simple Setting 3.
	:params args: dict of arguments from args.parser()
	"""
	def __init__(self, args):
		self.data_path = args.path +f'Data/own_data/setting1_3/dt_{args.n_train+args.n_val}_{args.n_cov}.csv'
		super().__init__(args)
	
	def x_fn(self):
		"""
		method to generate covariates
		"""
		self.x = multivariate_normal.rvs(mean=[0]*self.n_cov, cov=5, size=self.n)
		
	def t_obs_fn(self):
		"""
		method to generate treatments
		"""
		self.t_obs = np.expand_dims(chi2.rvs(df=2, size=self.n), axis=1)
		
	def t_opt_fn(self):
		"""
		method to generate optimal treatments
		"""
		alpha = np.array([1,.5,0.1])
		baseline = np.expand_dims(np.matmul(self.x, alpha), axis=1)
		#noise = np.expand_dims(norm.rvs(loc=0, scale = .1, size=self.n), axis=1)
		s = baseline# + noise
		self.t_opt = 5/(1+np.exp(-s))

	def y_obs_fn(self):
		"""
		method to generate outcomes
		"""
		diff = np.abs(self.t_opt-self.t_obs)
		self.y_obs = 2*np.exp(-diff/np.std(diff))
		
	def t_dens_obs_fn(self):
		"""
		method to generate conditional densities
		"""
		self.t_dens_obs = np.ones((self.n,1))
		
	def generate_value(self, x, t_obs, t_opt):
		"""
		method to generate the sum of counterfactual outcomes
		"""
		diff = (t_obs-t_opt).abs()
		value=2*(-diff/diff.std()).exp()
		return value.sum().item()

class setting2_1(own_sim_base):
	"""
	class for the simulation of Confounder Setting with linear effects.
	:params args: dict of arguments from args.parser()
	"""
	def __init__(self, args):
		self.data_path = args.path +f'Data/own_data/setting2_1/dt_{args.n_train+args.n_val}_{args.n_cov}.csv'
		super().__init__(args)
	
	def x_fn(self):
		"""
		method to generate covariates
		"""
		self.x = multivariate_normal.rvs(mean=[0]*self.n_cov, cov=5, size=self.n)
		
	def t_obs_fn(self):
		"""
		method to generate treatments
		"""
		self.alpha = np.array([1,.5,1])*self.alpha_factor
		baseline_x1 = np.expand_dims(np.matmul(self.x, self.alpha), axis=1)
		baseline_x2 = np.expand_dims(np.matmul(self.x**2, self.alpha*0.1), axis=1)
		
		sig = .5
		noise = np.expand_dims(norm.rvs(loc=0, scale = sig, size=self.n), axis=1)
		self.t_dens_obs_true = norm.pdf(noise, loc=0, scale=sig) +.05
		
		s = noise + baseline_x1 - baseline_x2 
		
		self.t_obs = 2/(1+np.exp(-s/np.std(s)*5))
		
	def t_opt_fn(self):
		"""
		method to generate optimal treatments
		"""
		self.gamma = np.array([.5,1,.5]) * self.gamma_factor
		self.gamma_torch = torch.Tensor(self.gamma)
		baseline_x_t = np.zeros(self.t_obs.shape)
		self.t_opt = baseline_x_t + self.beta_1/(self.beta_2*2)
		
	def y_obs_fn(self):
		"""
		method to generate outcomes
		"""
		self.delta = self.alpha * self.delta_factor
		self.delta_torch = torch.Tensor(self.delta)
		baseline_x = np.expand_dims(np.matmul(self.x, self.delta), axis=1)

		baseline_x_t = 0
		baseline_t = (self.beta_1+baseline_x_t) * self.t_obs - self.beta_2*np.power(self.t_obs,2)
		noise = np.expand_dims(norm.rvs(loc=0, scale = 0.25, size=self.n), axis=1)
		s = baseline_x + baseline_t + noise# + interaction# + nonlinear
		self.s_min, self.s_std = np.min(s), np.std(s)
		self.y_obs = (s - self.s_min)/self.s_std
		
	def t_dens_obs_fn(self):
		"""
		method to generate conditional densities
		"""
		if self.with_true_t_dens :
			self.t_dens_obs = self.t_dens_obs_true
		else: 
			self.t_dens_obs = np.ones(self.t_obs.shape)
		
	def generate_value(self, x, t_obs, t_opt):
		"""
		method to generate the sum of the counterfactual outcomes
		"""
		baseline_x = torch.matmul(x, self.delta_torch).unsqueeze(1)
		baseline_x_t = 0
		baseline_t = ((self.beta_1+baseline_x_t) * t_obs - self.beta_2*t_obs.pow(2))
		
		value = baseline_x + baseline_t
		return ((value-self.s_min)/self.s_std).sum().item()

class setting2_1a(own_sim_base):
	"""
	class for the simulation of Confounder Setting with interaction effects.
	:params args: dict of arguments from args.parser()
	"""
	def __init__(self, args):
		self.data_path = args.path +f'Data/own_data/setting2_1/dt_{args.n_train+args.n_val}_{args.n_cov}.csv'
		super().__init__(args)
	
	def x_fn(self):
		"""
		method to generate covariates
		"""
		self.x = multivariate_normal.rvs(mean=[0]*self.n_cov, cov=5, size=self.n)
		
	def t_obs_fn(self):
		"""
		method to generate treatments
		"""
		self.alpha = np.array([1,.5,1])*self.alpha_factor
		baseline_x1 = np.expand_dims(np.matmul(self.x, self.alpha), axis=1)
		baseline_x2 = np.expand_dims(np.matmul(self.x**2, self.alpha*0.1), axis=1)
		
		noise = np.expand_dims(norm.rvs(loc=0, scale = self.noise, size=self.n), axis=1)
		self.t_dens_obs_true = norm.pdf(noise, loc=0, scale=self.noise) +.05
		
		s = noise + baseline_x1 - baseline_x2 # + nonlinear
		
		self.t_obs = 2/(1+np.exp(-s/np.std(s)*5))
		
	def t_opt_fn(self):
		"""
		method to generate optimal treatments
		"""
		self.gamma = np.array([.5,1,.5]) * self.gamma_factor
		self.gamma_torch = torch.Tensor(self.gamma)
		baseline_x_t = np.expand_dims(np.matmul(self.x, self.gamma), axis=1)
		self.t_opt = baseline_x_t + self.beta_1/(self.beta_2*2)
		
	def y_obs_fn(self):
		"""
		method to generate outcomes
		"""
		self.delta = self.alpha * self.delta_factor
		self.delta_torch = torch.Tensor(self.delta)
		baseline_x = np.expand_dims(np.matmul(self.x, self.delta), axis=1)

		baseline_x_t = np.expand_dims(np.matmul(self.x, self.gamma), axis=1)
		baseline_t = (self.beta_1+baseline_x_t) * self.t_obs - self.beta_2*np.power(self.t_obs,2)
		noise = np.expand_dims(norm.rvs(loc=0, scale = 0.25, size=self.n), axis=1)
		s = baseline_x + baseline_t + noise
		self.y_obs = 3/(1+np.exp(-s/np.std(s)))
		
	def t_dens_obs_fn(self):
		"""
		method to generate conditional densities
		"""
		if self.with_true_t_dens :
			self.t_dens_obs = self.t_dens_obs_true
		else: 
			self.t_dens_obs = np.ones(self.t_obs.shape)
		
	def generate_value(self, x, t_obs, t_opt):
		"""
		method to generate the sum of the counterfactual outcomes
		"""
		baseline_x = torch.matmul(x, self.delta_torch).unsqueeze(1)
		baseline_x_t = torch.matmul(x, self.gamma_torch).unsqueeze(1)
		baseline_t = ((self.beta_1+baseline_x_t) * t_obs - self.beta_2*t_obs.pow(2))
		
		value = baseline_x + baseline_t
		return (3/(1+(-value/value.std()).exp())).sum().item()

class setting3_1(own_sim_base):
	"""
	class for the simulation of Complex Setting
	:params args: dict of arguments from args.parser()
	"""
	def __init__(self, args):
		self.data_path = args.path +f'Data/own_data/setting3_1/dt_{args.n_train+args.n_val}_{args.n_cov}.csv'
		super().__init__(args)
	
	def x_fn(self):
		"""
		method to generate covariates
		"""
		self.x = multivariate_normal.rvs(mean=[0]*self.n_cov, cov=5, size=self.n)
		
	def t_obs_fn(self):
		"""
		method to generate treatments
		"""
		self.alpha = np.array([1,.5,0])*self.alpha_factor
		baseline_x1 = np.expand_dims(np.matmul(self.x, self.alpha), axis=1)
		
		noise = np.expand_dims(norm.rvs(loc=0, scale = self.noise, size=self.n), axis=1)
		self.t_dens_obs_true = norm.pdf(noise, loc=0, scale=self.noise) + 0.05
		
		s = noise + baseline_x1
		
		self.t_obs = 2/(1+np.exp(-s/np.std(s)*5))

	def t_opt_fn(self):
		"""
		method to generate optimal treatments
		"""
		self.gamma = np.array([0,1,.5]) * self.gamma_factor
		self.gamma_torch = torch.Tensor(self.gamma)
		baseline_x_t = np.expand_dims(np.matmul(self.x, self.gamma), axis=1)
		self.t_opt = baseline_x_t + self.beta_1/(self.beta_2*2)

		
	def y_obs_fn(self):
		"""
		method to generate outcomes
		"""
		self.delta = np.array([0,.5,1]) * self.delta_factor
		self.delta_torch = torch.Tensor(self.delta)
		baseline_x = np.expand_dims(np.matmul(self.x, self.delta), axis=1)

		baseline_x_t = np.expand_dims(np.matmul(self.x, self.gamma), axis=1)

		baseline_t = (self.beta_1+baseline_x_t) * self.t_obs - self.beta_2*np.power(self.t_obs,2)
		noise = np.expand_dims(norm.rvs(loc=0, scale = 0.25, size=self.n), axis=1)
		s = baseline_x + baseline_t + noise
		self.y_obs = 3/(1+np.exp(-s/np.std(s)))
		
	def t_dens_obs_fn(self):
		"""
		method to generate conditional densities
		"""
		if self.with_true_t_dens :
			self.t_dens_obs = self.t_dens_obs_true
		else: 
			self.t_dens_obs = np.ones(self.t_obs.shape)
		
	def generate_value(self, x, t_obs, t_opt):
		"""
		method to generate the sum of the counterfactual outcomes
		"""
		baseline_x = torch.matmul(x, self.delta_torch).unsqueeze(1)
		baseline_x_t = torch.matmul(x, self.gamma_torch).unsqueeze(1)
		baseline_t = (self.beta_1+baseline_x_t) * t_obs - self.beta_1*t_obs.pow(2)
		
		value = baseline_x + baseline_t
		return (3/(1+(-value/value.std()).exp())).sum().item()

class TorchCategorizer():
	"""
	class to categorize the continuous treatment variable.
	"""
	def __init__(self, bins):
		self.bins = bins # how many categories do we want?
	
	def fit_transform(self, cont_data):
		"""
		fit categorizer and 
		"""
		categ_data, self.cutoffs=pd.qcut(cont_data.flatten(), q=self.bins, retbins=True, precision=2, labels=False)
		
		# save the means of each category to be able to turn categorical predictions back into continuous variables
		self.bin_means = [cont_data[categ_data == i].mean() for i in range(self.bins)] 
		new_cont_data = [self.bin_means[categ_data[i]] for i in range(len(categ_data))]
		
		return torch.Tensor(categ_data).unsqueeze(1), new_cont_data
	
	def transform(self, cont_data):
		categ_data = np.digitize(cont_data.flatten(), self.cutoffs[1:-1])
		new_cont_data = [self.bin_means[categ_data[i]] for i in range(len(categ_data))]
		return torch.Tensor(categ_data).unsqueeze(1), new_cont_data
	
	def inverse_transform(self, pred):
		categ_data = torch.max(pred, 1)[1]
		new_cont_data = [self.bin_means[categ_data[i]] for i in range(len(categ_data))]
		return torch.Tensor(new_cont_data).unsqueeze(1)
	
class TorchScaler():
	"""
	class to standardize inputs. Implemented same as sklearn scaler for torch tensors instead of numpy array
	"""
	def __init__(self, data):
		self.mean_ = data.mean(dim=0)
		self.std_ = data.std(dim=0)
		
	def transform(self, data):
		return (data - self.mean_) / self.std_
	
	def inverse_transform(self, data):
		return data * self.std_.to(data.device) + self.mean_.to(data.device)

def which_setting(args):
	"""
	function to initialize desired simulation setting with desired configurations (passed on with args)
	:params args: dict of arguments specified in args.parser
	"""
	if args.setting == '1_1':
		return setting1_1(args)
	elif args.setting == '1_1a':
		return setting1_1a(args)
	elif args.setting == '1_2':
		return setting1_2(args)
	elif args.setting == '1_3':
		return setting1_3(args)
	elif args.setting == '2_1':
		return setting2_1(args)
	elif args.setting == '2_1a':
		return setting2_1a(args)
	elif args.setting == '3_1':
		return setting3_1(args)
	else:
		print('No Setting chosen')
		exit()

