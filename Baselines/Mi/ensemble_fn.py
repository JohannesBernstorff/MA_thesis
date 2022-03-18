import time

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Model.net import *
from utils.general import *
from Baselines.Mi.train_fn import *
from Data.simulation_settings import *

class bagger_outcome():
	"""
	class to initialize, train, and bag the outcome model for the Mi-Baseline (first part)
	"""
	def __init__(self, dt, args):
		self.dt = dt
		self.criterion = nn.L1Loss(reduction='sum')
		
		self.initialize_base_model(args)
		self.set_params(args)

	def initialize_base_model(self, args):
		"""
		method to intialize the outcome model
		"""
		self.model_out = SimpleNet(d_in=args.n_cov, d_hidden=args.d_hidden_out, n_layer=args.n_layer_out, d_out=1, activation=args.activation, bn=args.bn, dropout=args.dropout, p_dropout=args.p_dropout)
		if torch.cuda.device_count() > 1 and args.num_gpu > 1:
			self.model_out = nn.DataParallel(self.model_out, device_ids=[0,1])
		self.model_out.to(self.dt.device)
		self.path_model_out = args.path+'Baselines/Mi/model_out/'

		torch.save(self.model_out.state_dict(), self.path_model_out+'initial.pth')

	def set_params(self, args):
		"""
		method to save the params from the argument-dictionary as class attributes
		"""
		self.lr, self.weight_decay, self.batch_size = args.lr, args.weight_decay, args.batch_size
		self.lr_scheduler, self.early_stopping, self.early_stopping_patience = args.lr_scheduler, args.early_stopping, args.early_stopping_patience
		self.verbose, self.write_tensorboard = args.verbose, args.write_tensorboard


	def initialize_fit_one(self):
		"""
		method to initialize one outcome model
		"""
		self.model_out.load_state_dict(torch.load(self.path_model_out+'initial.pth'))
		optimizer = torch.optim.Adam(self.model_out.parameters(), lr=self.lr, weight_decay=self.weight_decay)
		scheduler = None
		early_stopping = None
		if self.lr_scheduler:
			scheduler = ReduceLROnPlateau(
					optimizer, 
					mode='min',
					patience=50,
					cooldown=0,
					verbose=True)
		if self.early_stopping:
			early_stopping = EarlyStopping(patience=self.early_stopping_patience)
		return optimizer, scheduler, early_stopping

	def fit_one(self, n_epochs):
		"""
		method to fit one outcome model to the data
		"""
		optimizer, scheduler, early_stopping = self.initialize_fit_one()
		worker_out=worker_outcome(optim=optimizer, scheduler=scheduler, criterion=self.criterion)
		train_dataloader, val_dataloader = self.dt.build_dataloader_out_mi(batch_size=self.batch_size)

		for epoch in range(n_epochs):
			worker_out.train(self.model_out, train_dataloader, self.dt.device)
			worker_out.validate(self.model_out, val_dataloader, self.dt.device)
			
			if self.early_stopping:
				early_stopping(worker_out.loss/self.dt.n_val_boot)
				if early_stopping.early_stop:
					break

		loss_out_val=worker_out.evaluate(self.model_out, val_dataloader, self.dt.device)
		return loss_out_val/self.dt.n_val_boot

	def fit_bag(self, n_epochs, n_ensemble):
		"""
		method to fit bag of outcome models to the data
		"""
		self.bag_loss_val = np.empty(shape=n_ensemble)
		for i in range(n_ensemble):
			self.bag_loss_val[i] = self.fit_one(n_epochs)
			torch.save(self.model_out.state_dict(), self.path_model_out+str(i)+'.pth') # save individual model.state_dict, then we can predict ensembel
		return self.bag_loss_val
		
	def predict_ensemble_out(self):
		"""
		method to predict the dose/treatments with many models
		"""
		x_train, x_test= self.dt.return_X() 
		n_models = int(len(self.bag_loss_val)/2) # number of models used for predictions

		pred_train = torch.empty((self.dt.n_train+self.dt.n_val),n_models).to(self.dt.device) # empty tensor for predictions of the best models
		pred_test = torch.empty(self.dt.n_test,n_models).to(self.dt.device) # empty tensor for predictions of the best models

		for i, mod_ind in enumerate(self.bag_loss_val.argsort()[:n_models]):
			self.model_out.load_state_dict(torch.load(self.path_model_out+str(mod_ind)+'.pth')) # load the models
			self.model_out.eval()

			with torch.no_grad():
				x_train, x_test = x_train.to(self.dt.device), x_test.to(self.dt.device)
				pred_train[:,i]= self.model_out(x_train).flatten() 
				pred_test[:,i]= self.model_out(x_test).flatten()
		
		return pred_train.mean(dim=1, keepdim=True), pred_test.mean(dim=1, keepdim=True) # average over the individual model's predictions

class bagger_treatment():
	"""
	class to initialize, train, and bag the treatment prediction model for the Mi-Baseline (second part)
	"""
	def __init__(self, dt, args, scores, pred_train, pred_test):
		self.dt = dt
		self.criterion = nn.BCEWithLogitsLoss(reduction='none') # We use binary cross entropy loss
		
		self.initialize_base_model(args)
		self.set_params(args)

		self.pred_train, self.pred_test = pred_train, pred_test

	def initialize_base_model(self, args):
		"""
		method to intialize the treatment prediction model
		"""
		self.model_ITR = SimpleNet(d_in=args.n_cov, d_hidden=args.d_hidden_t, n_layer=args.n_layer_t, d_out=args.bins, activation=args.activation, bn=args.bn, dropout=args.dropout, p_dropout=args.p_dropout)
		if torch.cuda.device_count() > 1 and args.num_gpu > 1:
			self.model_ITR = nn.DataParallel(self.model_ITR, device_ids=[0,2])
		self.model_ITR.to(self.dt.device)
		self.path_model_ITR = args.path+'Baselines/Mi/model_ITR/'
		torch.save(self.model_ITR.state_dict(), self.path_model_ITR+'initial.pth')

	def set_params(self, args):
		"""
		method to save the parameters as attributes from the argument-dictionary. This facilitates handling of parameters thoughout the training procedure
		"""
		self.lr, self.weight_decay, self.batch_size = args.lr, args.weight_decay, args.batch_size
		self.lr_scheduler, self.early_stopping, self.early_stopping_patience = args.lr_scheduler, args.early_stopping, args.early_stopping_patience
		self.verbose, self.write_tensorboard = args.verbose, args.write_tensorboard

	def initialize_fit_one(self):
		"""
		method to intialize one treatment prediction model
		"""
		self.model_ITR.load_state_dict(torch.load(self.path_model_ITR+'initial.pth')) # we load the random state dict instead of sampling new, we tested that this does not make any difference in results but increases efficiency (less training time)
		
		optimizer = torch.optim.Adam(self.model_ITR.parameters(), lr=self.lr, weight_decay=self.weight_decay)
		scheduler = None
		early_stopping = None
		
		if self.lr_scheduler:
			scheduler = ReduceLROnPlateau(
					optimizer, 
					mode='min',
					patience=50,
					cooldown=0,
					verbose=True)
		if self.early_stopping:
			early_stopping = EarlyStopping(patience=self.early_stopping_patience)
		
		return optimizer, scheduler, early_stopping

	def fit_one(self, n_epochs):
		"""
		method to fit one treatment prediction model to the data
		"""
		optimizer, scheduler, early_stopping = self.initialize_fit_one()
		worker_treat=worker_treatment(cal_cf_y=self.dt.generate_value, optim=optimizer, scheduler=scheduler, criterion=self.criterion, categorizer=self.dt.categorizer)
		# we draw a bootstrap sample!
		train_dataloader, val_dataloader = self.dt.build_dataloader_ITR_mi(pred_train=self.pred_train, pred_test=self.pred_test, batch_size=self.batch_size)
		
		for epoch in range(n_epochs):
			worker_treat.train(self.model_ITR, train_dataloader, self.dt.device)
			worker_treat.validate(self.model_ITR, val_dataloader, self.dt.device)
			
			if self.early_stopping:
				early_stopping(worker_treat.loss/self.dt.n_val_boot)
				if early_stopping.early_stop:
					break

		loss_treat_val, value_treat_val=worker_treat.evaluate(self.model_ITR, val_dataloader, self.dt.device)
		return loss_treat_val/self.dt.n_val_boot

	def fit_bag(self, n_epochs, n_ensemble):
		"""
		method to fit a bag of treatment prediction models to the data
		:params n_epochs: number of epochs each model is supposed to be trained 
		:params n_ensemble: number of model that should be trained
		"""
		self.bag_loss_val = np.empty(shape=n_ensemble)

		for i in range(n_ensemble):
			self.bag_loss_val[i] = self.fit_one(n_epochs)
			torch.save(self.model_ITR.state_dict(), self.path_model_ITR+str(i)+'.pth') # save individual model so that we can use for prediction
		
		return self.bag_loss_val
	
	def evaluate_bag(self):
		"""
		method to evaluate the bag of models
		"""
		x_train, x_test= self.dt.return_X()
		n_models = int(len(self.bag_loss_val)/2) # how many models are used for the prediction

		pred_train = torch.empty((self.dt.n_train+self.dt.n_val), self.dt.bins, n_models).to(self.dt.device) # empty array for the predictions of each model 
		pred_test = torch.empty(self.dt.n_test, self.dt.bins, n_models).to(self.dt.device) # empty array for the predictions of each model 

		# load the best models and make predictions
		for i, mod_ind in enumerate(self.bag_loss_val.argsort()[:n_models]):
			self.model_ITR.load_state_dict(torch.load(self.path_model_ITR+str(mod_ind)+'.pth'))
			self.model_ITR.eval()
			with torch.no_grad():
				x_train, x_test = x_train.to(self.dt.device), x_test.to(self.dt.device)
				pred_train[:,:,i]= self.model_ITR(x_train)
				pred_test[:,:,i]= self.model_ITR(x_test)
		
		# We convert the categorical predictions back into continuous predictions
		pred_train = self.dt.categorizer.inverse_transform(pred_train.mean(dim=2))
		pred_test = self.dt.categorizer.inverse_transform(pred_test.mean(dim=2))
		
		# we calculate the value of each IDR
		t_opt_train, t_opt_test = torch.Tensor(self.dt.t_opt_train), torch.Tensor(self.dt.t_opt_test)
		value_train, value_test = self.dt.generate_value(x=x_train, t_obs=pred_train, t_opt=t_opt_train), self.dt.generate_value(x=x_test, t_obs=pred_test, t_opt=t_opt_test)
		return value_train/(self.dt.n_train+self.dt.n_val), value_test/self.dt.n_test

def fit_mi_repeat(args):
	"""
	function to fit Mi-Baseline repeatedly
	"""
	res = np.empty((args.n_repeats,2)) # where to save results

	if args.verbose:
		start = time.time()
		print(f'Mi: Start treatment model fitting: size: {args.n_train+args.n_val}')
	
	for i in range(args.n_repeats): # how often to repeat ensemble-fitting?
		if args.verbose:
			print(f'Reps: number {i} of {args.n_repeats}')

		dt = which_setting(args) # load data

		# fit bagged outcome model
		bagger_out = bagger_outcome(dt=dt, args=args)
		sol = bagger_out.fit_bag(n_epochs=args.n_epochs, n_ensemble=args.n_ensemble)
		pred_train, pred_test=bagger_out.predict_ensemble_out()

		# fit bagged treatment model
		bagger_treat = bagger_treatment(dt=dt, args=args, scores = sol, pred_train=pred_train, pred_test = pred_test)
		sol = bagger_treat.fit_bag(n_epochs=args.n_epochs, n_ensemble=args.n_ensemble)
		res[i,0], res[i,1] =bagger_treat.evaluate_bag()

	if args.verbose:
		end = time.time()
		print(f"Training time: {(end-start)/60:.3f} minutes")

	return {'mean_test': np.round(np.mean(res[:,1]), decimals=4),
		'std_test': np.round(np.std(res[:,1]), decimals=2),
		'mean_train': np.round(np.mean(res[:,0]), decimals=4),
		'std_train':  np.round(np.std(res[:,0]), decimals=2)}
