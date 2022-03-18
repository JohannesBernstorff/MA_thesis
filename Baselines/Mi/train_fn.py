import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Model.loss import *


class worker_outcome():
	"""
	class for the outcome model training procedure
	"""
	def __init__(self, optim, scheduler, criterion):
		self.optim = optim
		self.scheduler = scheduler
		self.criterion = criterion
		
	def train(self, model, train_dataloader, device):
		"""
		method for one training step 
		"""
		self.loss = 0
		model.train()
		for i, data in enumerate(train_dataloader):
			self.optim.zero_grad()
			X_s, y_s= data
			X_s, y_s = X_s.to(device), y_s.to(device)
			
			pred_s = model(X_s)
			
			loss = self.criterion(pred_s, y_s)
			loss.backward()

			self.optim.step()
			
			self.loss += loss.item()
			
	def validate(self, model, val_dataloader, device):
		"""
		method for one validation step 
		"""
		self.loss = 0
		model.eval()
		with torch.no_grad():
			for i, data in enumerate(val_dataloader):
				X_s, y_s = data
				X_s, y_s = X_s.to(device), y_s.to(device)
				
				pred_s = model(X_s)
				
				loss = self.criterion(pred_s,y_s)
				
				self.loss += loss.item()
		if self.scheduler != None:
			self.scheduler.step(self.loss)

	def evaluate(self, model, test_dataloader, device):
		"""
		method for one evaluation step 
		"""
		self.loss = 0
		model.eval()
		with torch.no_grad():
			for i, data in enumerate(test_dataloader):
				X_s, y_s = data
				X_s, y_s = X_s.to(device), y_s.to(device)

				pred_s= model(X_s)
				
				loss = self.criterion(pred_s, y_s)
				
				self.loss += loss.item()
		return self.loss
			
class worker_treatment():
	"""
	class for the treatment prediction model training procedure
	"""
	def __init__(self, cal_cf_y, optim, scheduler, criterion, categorizer):
		self.cal_cf_y = cal_cf_y
		self.optim = optim
		self.scheduler = scheduler
		self.criterion = criterion
		self.categorizer = categorizer

	def set_metrics(self):
		"""
		method to set counters for loss and value
		"""
		self.loss = 0
		self.value = 0
	
	def train(self, model, train_dataloader, device):
		"""
		method for one training step
		"""
		self.set_metrics()
		model.train()
		for i, data in enumerate(train_dataloader):
			self.optim.zero_grad()
			X_s, t_s, t_o_s, y_s = data
			X_s, t_s, t_o_s, y_s = X_s.to(device), t_s.to(device), t_o_s.to(device), y_s.to(device)
			
			pred_owl_s = model(X_s)
			
			# the loss in Mi-Baseline can simply be divided by the residuals, t_s.
			loss = self.criterion(pred_owl_s, t_s)
			weighted_loss=torch.mul(input=loss,other=y_s).sum()
			weighted_loss.backward()

			self.optim.step()

			self.loss += loss.sum().item()
			
			pred_owl_s = self.categorizer.inverse_transform(pred_owl_s)
			self.value += self.cal_cf_y(x=X_s, t_obs=pred_owl_s.to(device), t_opt=t_o_s)
	
	def validate(self, model, val_dataloader, device):
		"""
		method for one validation step
		"""
		self.set_metrics()
		model.eval()
		with torch.no_grad():
			for i, data in enumerate(val_dataloader):
				X_s, t_s, t_o_s, y_s = data
				X_s, t_s, t_o_s, y_s = X_s.to(device), t_s.to(device), t_o_s.to(device), y_s.to(device)
				
				pred_owl_s = model(X_s)
				
				loss = self.criterion(pred_owl_s, t_s)
			
				self.loss += loss.sum().item()

				pred_owl_s = self.categorizer.inverse_transform(pred_owl_s)
				self.value += self.cal_cf_y(x=X_s, t_obs=pred_owl_s.to(device), t_opt=t_o_s)
		if self.scheduler != None:
			self.scheduler.step(self.loss)

	def evaluate(self, model, test_dataloader, device):
		"""
		method for one evaluation step
		"""
		self.set_metrics()
		model.eval()
		with torch.no_grad():
			for i, data in enumerate(test_dataloader):
				X_s, t_s, t_o_s, y_s = data
				X_s, t_s, t_o_s, y_s = X_s.to(device), t_s.to(device), t_o_s.to(device), y_s.to(device)
				
				pred_owl_s = model(X_s)

				loss = self.criterion(pred_owl_s, t_s)
			
				self.loss += loss.sum().item()

				pred_owl_s = self.categorizer.inverse_transform(pred_owl_s)
				self.value += self.cal_cf_y(x=X_s, t_obs=pred_owl_s.to(device), t_opt=t_o_s)
		return self.loss, self.value

