import torch
import torch.nn as nn
import torch.nn.functional as F

class my_loss_detached_I(nn.Module):
	"""
	own class inheriting from nn.Module and implements detached indicator loss function
	"""
	def __init__(self, y_pow=1, phi = 3):
		super(my_loss_detached_I, self).__init__()
		"""
		:params y_pow: outcome raised to the power of y_pow wihtin each batch
		:params phi: range around actual outcome
		"""
		self.y_pow = y_pow
		self.phi = phi

	def forward(self, pred, actual, ps, y):
		loss = F.l1_loss(pred, actual, reduction = 'none') 
		I_phi=torch.where(loss/self.phi>1,loss/self.phi, torch.ones(loss.size(), device=loss.device)) 
		I_with_gradient=torch.div(input=loss,other=I_phi.detach()) # scale to one outside of phi while keeping gradients intact
		
		y = (y/y.mean()).pow(self.y_pow) # raise outcome to the power of y_pow
		w = torch.div(input=y,other=ps) # divide by conditional density
		
		loss = torch.mul(input=I_with_gradient,other=w)
		return loss.sum()

	def step(self):
		return None

class my_loss_surrogate_I(nn.Module):
	"""
	own class inheriting from nn.Module and implements surrogate indicator loss function
	"""
	def __init__(self, y_pow=1, phi = 3):
		super(my_loss_surrogate_I, self).__init__()
		"""
		:params y_pow: outcome raised to the power of y_pow wihtin each batch
		:params phi: range around actual outcome
		"""
		self.y_pow = y_pow
		self.phi = phi

	def forward(self, pred, actual, ps, y):
		loss = F.l1_loss(pred, actual, reduction = 'none')
		smooth_I_loss = 1 - torch.exp(-loss/self.phi) + 0.05 * loss # surrogate indicator function
		
		y = (y/y.mean()).pow(self.y_pow) # raise outcome to the power of y_pow
		w = torch.div(input=y,other=ps) # divide by conditional density

		loss = torch.mul(input=smooth_I_loss,other=w)
		return loss.sum()

	def step(self):
		return None

class my_loss_abs_dev(nn.Module):
	"""
	own class inheriting from nn.Module and implements absolute deviation loss function
	"""
	def __init__(self, y_pow=1, phi=1):
		super(my_loss_abs_dev, self).__init__()
		"""
		:params y_pow: outcome raised to the power of y_pow wihtin each batch
		:params phi: range around actual outcome
		"""
		self.y_pow = y_pow

	def forward(self, pred, actual, ps, y):
		loss = F.l1_loss(pred, actual, reduction = 'none')
		y = (y/y.mean()).pow(self.y_pow) # raise outcome to the power of y_pow
		w = torch.div(input=y,other=ps) # divide by conditional density
		loss = torch.mul(input=loss,other=w)
		return loss.sum()

	def step(self):
		return None

# Dynamic Loss Extension 1:
# loss_phase1 is simple absolute deviation weighed with outcome and conditional density
# loss_phase2 is proposed loss function, either detached indicator or surrogate indicator
class my_loss_detached_I_dyn1(nn.Module):
	"""
	own class inheriting from nn.Module and implements detached indicator loss function with the dynamic extension 1:
	"""
	def __init__(self, y_pow=1, phi = .5, phase1_end=10):
		super(my_loss_detached_I_dyn1, self).__init__()
		"""
		:params y_pow: outcome raised to the power of y_pow wihtin each batch
		:params phi: range around actual outcome
		:params phase1_end: at what epoch should the objective be solely the detached indicator
		"""
		self.y_pow = y_pow
		self.phi = phi
		
		self.loss_ratio = 0 # at initialization only loss_phase1
		self.phase=phase1_end
		self.turning_point=phase1_end # at what epoch should the objective only be influenced by the detached indicator

	def forward(self, pred, actual, ps, y):
		loss = F.l1_loss(pred, actual, reduction = 'none')
		I_phi=torch.where(loss/self.phi>1,loss/self.phi, torch.ones(loss.size(), device=loss.device))
		I_with_gradient=torch.div(input=loss,other=I_phi.detach()) # scale to one outside of phi while keeping gradients intact
		
		y = (y/y.mean()).pow(self.y_pow) # raise outcome to the power of y_pow
		w = torch.div(input=y,other=ps) # divide by conditional density
		
		loss_phase1 = torch.mul(input=loss,other=w).sum() # usual absolute deviations loss weighed by outcome and conditional density
		loss_phase2 = torch.mul(input=I_with_gradient,other=w).sum() 
		loss = (1-self.loss_ratio) * loss_phase1 + self.loss_ratio * loss_phase2
		return loss

	def step(self):
		# similiar to pytorch optimizer class we define step function for the criterions that control the dynamics of the loss functions
		self.loss_ratio = min(1,(self.turning_point-self.phase)/self.turning_point) 
		self.phase -=1

class my_loss_surrogate_I_dyn1(nn.Module):
	def __init__(self, y_pow=1, phi = .5, phase1_end=40):
		super(my_loss_surrogate_I_dyn1, self).__init__()
		"""
		:params y_pow: outcome raised to the power of y_pow wihtin each batch
		:params phi: range around actual outcome
		:params phase1_end: at what epoch should the objective be solely the surrogate indicator
		"""
		self.y_pow = y_pow
		self.phi = phi

		self.loss_ratio = 0 # at first only the absolute deviations loss and not the surrogate influence the objectve
		self.phase=phase1_end 
		self.turning_point=phase1_end # at what epoch should the objective only be influenced by the surrogate indicator

	def forward(self, pred, actual, ps, y):
		loss = F.l1_loss(pred, actual, reduction = 'none')
		smooth_I_loss = 1 - torch.exp(-loss/self.phi) + 0.05 * loss # surrogate loss function for the indiciator
		
		y = (y/y.mean()).pow(self.y_pow) # raise outcome to the power of y_pow
		w = torch.div(input=y,other=ps) # divide by conditional density
		
		loss_phase1 = torch.mul(input=loss,other=w).sum() # usual absolute deviations loss weighed by outcome and conditional density
		loss_phase2 = torch.mul(input=smooth_I_loss,other=w).sum()
		loss = (1-self.loss_ratio) * loss_phase1 + self.loss_ratio * loss_phase2
		return loss

	def step(self):
		# similiar to pytorch optimizer class we define step function for the criterions that control the dynamics of the loss functions
		self.loss_ratio = min(1,(self.turning_point-self.phase)/self.turning_point)
		self.phase -=1

# Dynamic Loss Extension 2:
# phi narrows down over training
# first phi is set to max_phi and decreases to phi		
class my_loss_detached_I_dyn2(nn.Module):
	def __init__(self, y_pow=1, phi = .5, max_phi=5, phase1_end=20):
		super(my_loss_detached_I_dyn2, self).__init__()
		"""
		:params y_pow: outcome raised to the power of y_pow wihtin each batch
		:params phi: range around actual outcome
		:params max_phi: initial value of phi, usually set to a wide range
		:params phase1_end: at what epoch should the objective be solely the detached indicator
		"""
		self.y_pow = y_pow
		
		self.phi_step = (max_phi - phi)/phase1_end # how much does phi decrease in every epoch
		self.phi = max_phi # we set the initial phi to the maximum phi 
		self.phase = phase1_end # at what epoch should phi change no more

	def forward(self, pred, actual, ps, y):
		loss = F.l1_loss(pred, actual, reduction = 'none')
		
		I_phi=torch.where(loss/self.phi>1,loss/self.phi, torch.ones(loss.size(), device=loss.device))
		I_with_gradient=torch.div(input=loss,other=I_phi.detach()) # scale to one outside of phi while keeping gradients intact
		
		y = (y/y.mean()).pow(self.y_pow) # raise outcome to the power of y_pow
		w = torch.div(input=y,other=ps) # divide by conditional density
		loss = torch.mul(input=I_with_gradient,other=w).sum()
		return loss

	def step(self):
		# similiar to pytorch optimizer class we define step function for the criterions that control the dynamics of the loss functions
		if self.phase > 0: 
			self.phi -= self.phi_step
		self.phase -= 1

class my_loss_surrogate_I_dyn2(nn.Module):
	def __init__(self, y_pow=1, phi = .5, max_phi=10, phase1_end=20):
		super(my_loss_surrogate_I_dyn2, self).__init__()
		"""
		:params y_pow: outcome raised to the power of y_pow wihtin each batch
		:params phi: range around actual outcome
		:params max_phi: initial value of phi, usually set to a wide range
		:params phase1_end: at what epoch should the objective be solely the detached indicator
		"""
		self.y_pow = y_pow
		
		self.phi_step = (max_phi - phi)/phase1_end # how much does phi decrease in every epoch
		self.phi = max_phi # we set the initial phi to the maximum phi 
		self.phase = phase1_end # at what epoch should phi change no more
		
	def forward(self, pred, actual, ps, y):
		loss = F.l1_loss(pred, actual, reduction = 'none')
		smooth_I_loss = 1 - torch.exp(-loss/self.phi) + 0.05 * loss # surrogate loss function for the indicator 
		
		y = (y/y.mean()).pow(self.y_pow) # raise outcome to the power of y_pow
		w = torch.div(input=y,other=ps) # divide by conditional density
		
		loss = torch.mul(input=smooth_I_loss,other=w).sum()
		return loss

	def step(self):
		# similiar to pytorch optimizer class we define step function for the criterions that control the dynamics of the loss functions
		if self.phase > 0: 
			self.phi -= self.phi_step
		self.phase -= 1


def initialize_criterion(crit_name, args):
	"""
	function to initialize desired loss function with desired configurations (passed on with args)
	:params crit_name: string with name of desired criterion
	:params args: dict of arguments specified in args.parser
	"""
	if crit_name == "my_loss_detached_I":
		return my_loss_detached_I(y_pow=args.y_pow, phi=args.phi)

	elif crit_name == "my_loss_surrogate_I":
		return my_loss_surrogate_I(y_pow=args.y_pow, phi=args.phi)

	elif crit_name == "my_loss_abs_dev":
		return my_loss_abs_dev( y_pow=args.y_pow, phi=args.phi)

	elif crit_name == "my_loss_surrogate_I_dyn1":
		return my_loss_surrogate_I_dyn1(y_pow=args.y_pow, phi=args.phi, phase1_end=args.phase1_end)

	elif crit_name == "my_loss_detached_I_dyn1":
		return my_loss_detached_I_dyn1(y_pow=args.y_pow, phi=args.phi, phase1_end=args.phase1_end)

	elif crit_name == "my_loss_detached_I_dyn2":
		return my_loss_detached_I_dyn2(y_pow=args.y_pow, phi=args.phi, max_phi=args.max_phi, phase1_end=args.phase1_end)

	elif crit_name == "my_loss_surrogate_I_dyn2":
		return my_loss_surrogate_I_dyn2(y_pow=args.y_pow, phi=args.phi, max_phi=args.max_phi, phase1_end=args.phase1_end)

	else:
		print('ERROR at loss function input')
		exit()

class my_l1(nn.Module):
	"""
	Own implementation of absolute deviation loss
	necessary because additional arguments will be passed when used in training and standard pytorch implementation does not support additional arguments
	"""
	def __init__(self, y_pow=1, phi=1):
		super(my_l1, self).__init__()
		"""
		:params y_pow: outcome raised to the power of y_pow wihtin each batch
		:params phi: range around actual outcome
		"""

	def forward(self, pred, actual, ps, y):
		loss = F.l1_loss(pred, actual, reduction = 'none')
		return loss

	def step(self):
		return None


class my_l2(nn.Module):
	"""
	Own implementation of mean squared error loss
	necessary because additional arguments will be passed when used in training and standard pytorch implementation does not support additional arguments
	"""
	def __init__(self, y_pow=1, phi=1):
		super(my_l2, self).__init__()
		"""
		:params y_pow: outcome raised to the power of y_pow wihtin each batch
		:params phi: range around actual outcome
		"""

	def forward(self, pred, actual, ps, y):
		loss = F.mse_loss(pred, actual, reduction = 'none')
		return loss

	def step(self):
		return None
