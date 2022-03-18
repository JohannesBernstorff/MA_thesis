import math
import numpy as np

from scipy.stats import norm

import torch

def std_norm_density_custom(e):
	"""
	Calculate standard normal density
	"""
	return 1 / np.sqrt(2 * math.pi) * (-.5 * e.pow(2)).exp()

def trunc_norm_density_custom(sample, mu, sig, lower=0, upper=2):
	"""
	Calculate truncated normal density
	"""
	n_dens_samp = std_norm_density_custom((sample-mu)/sig)
	n_cum_lower = norm.cdf((lower-mu.cpu().numpy())/sig)
	n_cum_upper = norm.cdf((upper-mu.cpu().numpy())/sig)
	truncnorm_dens = (1/sig) * (n_dens_samp/torch.Tensor(n_cum_upper-n_cum_lower).to('cuda'))
	return truncnorm_dens

def dynamic_weight_fn(epoch, n_epochs, w_type='grl'):
	"""
	function to calculate weights that change over the training process
	:params epoch: at which epoch to calculate weight
	:params n_epoch: number of epochs in total
	:params w_type: which type of dynamic regime
	"""
	if w_type == 'grl':
		# smooth function from 0 at first epoch to 1 at last epoch
		# controls weighting of gradient reversal and phi over training process
		dynamic_weight = 2. / (1. + np.exp(-10. * float(epoch)/n_epochs)) - 1.
		return dynamic_weight
	elif w_type == 'owl': # smooth function from 1 to 0
		dynamic_weight = 1. / (1. + 2.* np.exp(10. * (float(epoch)/n_epochs-.5)))
		return dynamic_weight

def scale_weights_fn(loss_owl, loss_adv):
	"""
	function to calculate balancing weights between the loss functions
	"""
	l_o, l_a = loss_owl.detach(), loss_adv.detach()
	return loss_owl * (l_a/l_o)

class EarlyStopping():
	"""
	Early stopping to stop the training when the loss does not improve after certain epochs.
	"""
	def __init__(self, patience=30, min_delta=0):
		"""
		:param patience: how many epochs to wait before stopping when loss is not improving
		:param min_delta: minimum difference between new loss and old loss for new loss to be considered as an improvement
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.best_loss = None
		self.early_stop = False
		
	def __call__(self, val_loss):
		if self.best_loss == None:
			self.best_loss = val_loss
		elif self.best_loss - val_loss > self.min_delta:
			self.best_loss = val_loss
			self.counter = 0
		elif self.best_loss - val_loss < self.min_delta:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True