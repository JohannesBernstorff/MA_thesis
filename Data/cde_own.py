import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from Data.DeepCDE.deepcde.bases.cosine import CosineBasis
from Data.DeepCDE.deepcde.deepcde_pytorch import cde_layer, cde_loss, cde_predict
from Data.DeepCDE.deepcde.utils import box_transform


class CDENet(nn.Module):
	""" 
	Neural network class for conditional density estimation
	"""
	def __init__(self, basis_size, n_cov=9):
		""" 
		:params basis_size: number of basis expansions for nonparametric density estimation
		:params n_cov: number of covariates 
		"""
		super(CDENet, self).__init__()
		self.l1 = nn.Linear(n_cov, 30)
		self.l2 = nn.Linear(30, 30)
		self.l3 = nn.Linear(30, 30)
		self.cde = cde_layer(30, basis_size)    # CDE layer here

		self.dropout = nn.Dropout(0.3)
	
	def forward(self, x):
		x = F.leaky_relu(self.l1(x))
		x = F.leaky_relu(self.l2(x))
		x = self.dropout(x)
		x = F.leaky_relu(self.l3(x))
		beta = self.cde(x)
		return beta

def run_cde(x_train_val, t_obs_train_val):
	"""
	function to run conditional density estimation with neural networks
	:params x_train_val: covariate matrix used for estimation
	:params t_obs_train_val: continuous treatment variable, is target variable
	"""
	n_train = t_obs_train_val.shape[0]
	n_basis = 30 # we pick 30-dimensional basis expansion
	basis = CosineBasis(n_basis) # we use the cosine basis for the basis expansion

	train_ind, val_ind = train_val_split(n_train)
	x_train, x_val, x_train_val, scaler = build_X(x_train_val, train_ind, val_ind)
	
	min_val = min(t_obs_train_val) # not correct but otherwise box_tranform function breaks
	max_val = max(t_obs_train_val) # not correct but otherwise box_tranform function breaks
	
	z_train = box_transform(t_obs_train_val[train_ind], min_val, max_val)  # transform to a variable between 0 and 1
	z_val = box_transform(t_obs_train_val[val_ind], min_val, max_val)  # transform to a variable between 0 and 1
	z_train_val = box_transform(t_obs_train_val, min_val, max_val)  # transform to a variable between 0 and 1
	
	z_basis = basis.evaluate(z_train)[:, 1:] # evaluate basis, remove first coefficient
	z_val_basis = basis.evaluate(z_val)[:, 1:]  # evaluate basis, remove first coefficient
	
	z=torch.from_numpy(z_basis.astype(np.float64)).type(torch.Tensor)
	z_val=torch.from_numpy(z_val_basis.astype(np.float64)).type(torch.Tensor)

	train_dataloader = build_dataloader_cde(x_train, z) # we built a separate dataloder function for the density estimation 
	
	model = CDENet(basis_size=n_basis, n_cov=x_train_val.shape[1])
	custom_loss = cde_loss() # special cde_loss proposed by DeepCDE method
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

	device = set_device()
	model, x_val, z_val= model.to(device), x_val.to(device), z_val.to(device)
		
	n_epochs = 500
	epoch_check = 25 # every 25 epochs we check whehter the validation loss decreased
	
	loss_list = []
	valid_loss_list = []

	for epoch in range(n_epochs): # training procedure
		model.train()
		for batch_idx, (x_batch, z_basis_batch) in enumerate(train_dataloader):
			x_batch, z_basis_batch = x_batch.to(device), z_basis_batch.to(device)
			optimizer.zero_grad()
			beta_batch = model(x_batch)
			loss = custom_loss(beta_batch, z_basis_batch) 
			loss.backward()
			optimizer.step()
			loss_list.append(loss.item()) 
		
		if epoch % epoch_check == 0:
			beta_pred_val = model(x_val)
			loss_valid = custom_loss(beta_pred_val, z_val).item()
			print('Epoch %d, Training Loss: %.3f, Validation Loss: %.3f' %(epoch, loss_list[-1], loss_valid))

			valid_loss_list.append(loss_valid)
			if len(valid_loss_list) > 2 and (valid_loss_list[-1] > valid_loss_list[-2]):
				break

	# PREDICT density with fully trained model
	n_grid = 1000
	z_grid = np.linspace(0, 1, n_grid)  # Creating a grid over the density range
	
	x_train_val = x_train_val.to(device)
	beta_prediction_train_val = model(x_train_val)
	cde_train_val = cde_predict(model_output=beta_prediction_train_val.cpu().detach().numpy(),
				z_min=0, z_max=1, z_grid=z_grid, basis=basis)
	
	dens = np.empty(t_obs_train_val.shape)
	for ii, obs in enumerate(z_train_val):
		ind = np.where(np.logical_and(z_grid>=obs-0.1, z_grid<=obs+0.1))
		dens[ii] = np.mean(cde_train_val[ii,ind])
	return dens
	''' CHECK whether the predicted densities are sensible
	fig = plt.figure(figsize=(30, 20))
	for jj, cde_predicted in enumerate(cde_train_val[:12,:]):
		ax = fig.add_subplot(3, 4, jj + 1)
		plt.plot(z_grid, cde_predicted, label=r'$\hat{p}(z| x_{\rm obs})$')
		plt.axvline(z_train_val[jj], color='red', label=r'$z_{\rm obs}$')
		plt.xticks(size=16)
		plt.yticks(size=16)
		plt.xlabel(r'Redshift $z$', size=20)
		plt.ylabel('CDE', size=20)
		plt.legend(loc='upper right', prop={'size': 20})
	plt.savefig(fname=f'/local/home/jbernstorff/repo/adversarialrandomization/Code/IDR/Data/fig')
	exit()
	'''

def train_val_split(n, split=0.1):
	"""
	function for training-validation split in density estimation
	"""
	val_ind = np.random.choice(n, size=int(n*split), replace=False)
	train_ind = np.array([x for x in range(n) if x not in val_ind])
	return train_ind, val_ind

def build_X(x_train_val, train_ind, val_ind):
	"""
	function to build, subset and rescale X for density estimation
	"""
	x_train, x_val, x_train_val =torch.Tensor(x_train_val[train_ind,:]), torch.Tensor(x_train_val[val_ind,:]), torch.Tensor(x_train_val)
	scaler=TorchScaler(x_train)
	x_train=scaler.transform(x_train)
	x_val=scaler.transform(x_val)
	x_train_val=scaler.transform(x_train_val)
	return x_train, x_val, x_train_val, scaler

def build_dataloader_cde(x_train, z_train, batch_size = 50):
	"""
	function to build dataloader for density estimation
	"""
	train_dataset=TensorDataset(x_train, z_train)
	train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
	return train_dataloader

def set_device():
	"""
	function to set device for density estimation
	"""
	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda"
	return device

	
class TorchScaler():
	"""
	class for standardizing X. Same as in simulation_settings.py. (Inspired from sklearn-version, but for tensors)
	"""
	def __init__(self, data):
		self.mean_ = data.mean(dim=0)
		self.std_ = data.std(dim=0)
		
	def transform(self, data):
		return (data - self.mean_) / self.std_
	
	def inverse_transform(self, data):
		return data * self.std_.to(data.device) + self.mean_.to(data.device)
