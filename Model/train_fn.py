import torch
#import torch.nn as nn
#from torch.autograd import Function
#import torch.nn.functional as F
#from torch.utils.data import DataLoader

from Model.loss import *
from Data.simulation_settings import *

class worker_owl():
	"""
	class to implement the training procedure of our neural networks
	"""
	def __init__(self, cal_cf_y, optim, scheduler, criterion_owl, criterion_adv, AdvRand = False, grl_weight = 0.5):
		"""
		:params cal_cf_y: pass function to generate counterfactual outcomes. Not used in training or validation! Simply used for reference and evaluation!
		:params optim: pass optimizer pytorch class, mostly used ADAM
		:params scheduler: pass learning rate scheduler class from pytorch's optim
		:params criterion_owl: pass loss function class for the OWL objective
		:params criterion_adv: pass loss function for the second objective (the adversary)
		:params AdvRand: pass True/False to indicate whehter to use adversarial randomization
		:params grl_weight: weight parameter in the gradient reversal layer. Controls the influence of the adversary on the feature extractor network
		"""
		self.cal_cf_y = cal_cf_y
		self.optim = optim
		self.scheduler = scheduler
		self.criterion_owl = criterion_owl
		self.criterion_adv = criterion_adv
		self.AdvRand, self.grl_weight = AdvRand, grl_weight

	def set_metrics(self):
		""" 
		counters for each epoch 
		"""
		self.loss_owl = 0 
		self.loss_adv = 0
		self.value_owl = 0
		self.value_adv = 0
	
	def train(self, model, train_dataloader, device):
		"""
		function for one training step (one epoch)
		"""
		self.set_metrics() # set counters to zero
		model.train()
		for i, data in enumerate(train_dataloader):
			self.optim.zero_grad()
			# t_s is the treatment batch, t_o_s is the optimal treatment batch
			# y_s is the outcome batch, t_d_s is the conditional density of treatment given X
			X_s, t_s, t_o_s, y_s, t_d_s = data 
			X_s, t_s, t_o_s, y_s, t_d_s = X_s.to(device), t_s.to(device), t_o_s.to(device), y_s.to(device), t_d_s.to(device)
			
			if self.AdvRand:
				# pred_owl_s is the prediction of the OWL network, pred_adv_s is the prediction of the adversary
				pred_owl_s, pred_adv_s = model(X_s, grl_lambda = self.grl_weight) 
			else:
				pred_owl_s, pred_adv_s = model(X_s, grl_lambda = 0)

			loss_adv = self.criterion_adv(pred=pred_adv_s, actual=t_s, ps=t_d_s, y=y_s)
			if self.AdvRand:
				w_adv = 1/((-loss_adv.detach()*0.5).exp()+0.1) # we calculate the "conditional density" for the OWL-objective based on the adversary
				# above calculation is inspired loosely by the normal density function
				# the higher the difference between predicted and observed treatment, the lower the "conditional density"!
				# we detach the loss, the conditional density is not followed in the backpropagation algorithm
				loss_owl = self.criterion_owl(pred=pred_owl_s, actual=t_s, ps=w_adv/w_adv.mean(), y=y_s) 
			else:
				loss_owl = self.criterion_owl(pred=pred_owl_s, actual=t_s, ps=t_d_s, y=y_s)
		
			loss = loss_owl + loss_adv.sum()
			
			loss.backward()

			self.optim.step()

			self.loss_owl += loss_owl.item()
			self.loss_adv += loss_adv.sum().item()
			# Below does NOT influence the training objective: 
			self.value_owl += self.cal_cf_y(x=X_s, t_obs=pred_owl_s, t_opt=t_o_s) # calculate the counterfactual outcomes and value of the IDR
			self.value_adv += self.cal_cf_y(x=X_s, t_obs=pred_adv_s, t_opt=t_o_s) # calculate the counterfactual outcomes
		self.criterion_owl.step() # need the step on the criterion for the dynamic loss functions
		return self.loss_owl, self.loss_adv, self.value_owl, self.value_adv
	
	def validate(self, model, val_dataloader, device):
		"""
		function for one validaiton step
		"""
		self.set_metrics() # set counters to zero
		model.eval()
		with torch.no_grad():
			for i, data in enumerate(val_dataloader):
				# t_s is the treatment batch, t_o_s is the optimal treatment batch
				# y_s is the outcome batch, t_d_s is the conditional density of treatment given X
				X_s, t_s, t_o_s, y_s, t_d_s = data
				X_s, t_s, t_o_s, y_s, t_d_s = X_s.to(device), t_s.to(device), t_o_s.to(device), y_s.to(device), t_d_s.to(device)
				
				# pred_owl_s is the prediction of the OWL network, pred_adv_s is the prediction of the adversary
				pred_owl_s, pred_adv_s = model(X_s, grl_lambda = 1)
				
				loss_owl = self.criterion_owl(pred=pred_owl_s, actual=t_s, ps=t_d_s, y=y_s)
				loss_adv = self.criterion_adv(pred=pred_adv_s, actual=t_s, ps=t_d_s, y=y_s).sum()
			
				self.loss_owl += loss_owl.item()
				self.loss_adv += loss_adv.item()
				# below does NOT influence the training objective: 
				self.value_owl += self.cal_cf_y(x=X_s, t_obs=pred_owl_s, t_opt=t_o_s) # calculate the counterfactual outcomes and value of the IDR
				self.value_adv += self.cal_cf_y(x=X_s, t_obs=pred_adv_s, t_opt=t_o_s) # calculate the counterfactual outcomes
		
		if self.scheduler != None: # one step of the learning rate scheduler
			self.scheduler.step(self.loss_owl)
		
		return self.loss_owl, self.loss_adv, self.value_owl, self.value_adv

'''
	def evaluate(self, model, test_dataloader, device):
		self.set_metrics()
		model.eval()
		with torch.no_grad():
			for i, data in enumerate(test_dataloader):
				X_s, t_s, t_o_s, y_s, t_d_s = data
				X_s, t_s, t_o_s, y_s, t_d_s = X_s.to(device), t_s.to(device), t_o_s.to(device), y_s.to(device), t_d_s.to(device)

				pred_owl_s, pred_adv_s = model(X_s, grl_lambda = 1)
				
				loss_owl = self.criterion_owl(pred=pred_owl_s, actual=t_s, ps=t_d_s, y=y_s)
				loss_adv = self.criterion_adv(pred=pred_adv_s, actual=t_s, ps=t_d_s, y=y_s).sum()
			
				self.loss_owl += loss_owl.item()
				self.loss_adv += loss_adv.item()
				# below does NOT influence the training objective: 
				self.value_owl += self.cal_cf_y(x=X_s, t_obs=pred_owl_s, t_opt=t_o_s)
				self.value_adv += self.cal_cf_y(x=X_s, t_obs=pred_adv_s, t_opt=t_o_s)
		return self.loss_owl, self.loss_adv, self.value_owl, self.value_adv
'''

def evaluate_ensemble(model, dt, path_model, sorted_scores, device):
	"""
	function to evaluate an ensemble of models
	"""
	
	n_models = int(len(sorted_scores)/2)  # half of the models are chosen for evaluation

	pred_train = torch.empty(dt.n_train+dt.n_val,1, n_models).to(device) # where to save the predictions of the training observations
	pred_test = torch.empty(dt.n_test,1, n_models).to(device) # where to save the predictions of the testing observations 

	x_train, x_test = torch.Tensor(dt.x_train).to(device), torch.Tensor(dt.x_test).to(device) # create covariate matrix for training and testing data

	# We are not using the dataloaders anymore, because we want the predictions in clear order. Hence we need to scale the inputs
	if dt.scale_dt:
		dt.scaler=TorchScaler(x_train) # We use the scaler class defined in the simulation_settings.py
		x_train=dt.scaler.transform(x_train)
		x_test=dt.scaler.transform(x_test)

	for i, mod_ind in enumerate(sorted_scores[:n_models]):
		# for the best n models we load the models' weights and biases
		# then we predict the treatments for the best models
		model.load_state_dict(torch.load(path_model+str(mod_ind)+'.pth'))
		model.eval()
		with torch.no_grad():
			# we fill the matrices with the predicted treatments of the observations
			pred_train[:,:,i], _ = model(x_train, grl_lambda = 1) 
			pred_test[:,:,i], _ = model(x_test, grl_lambda = 1)

	# we calculate the counterfactual value of the predicted treatments' mean
	value_train=dt.generate_value(x=x_train, t_obs=pred_train.mean(dim=2), t_opt=dt.t_opt_train)
	value_test=dt.generate_value(x=x_test, t_obs=pred_test.mean(dim=2), t_opt=dt.t_opt_test)
	return value_train, value_test