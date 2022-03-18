import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Model.loss import *

def activation_func(activation):
	# function to build activation function
	# returns nn.Module of activation function
	# we experimented with all activation funcitons below but achieve the best results with elu
	# with simpler networks the activation functions tanh and sigmoid are good as well
	dict_mod = nn.ModuleDict([
		['elu', nn.ELU()],
		['tanh', nn.Tanh()],
		['sigmoid', nn.Sigmoid()],
		['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
		['selu', nn.SELU()],
		['none', nn.Identity()]])
	return  dict_mod[activation]


def LinearBlock(d_in, d_out, bn=True, dropout=True, activation='elu', p_dropout=.1):
	# function to build linear blocks with optional batch normalization and dropout 
	activations = activation_func(activation)
	
	block = [nn.Linear(d_in, d_out)]

	block.append(nn.BatchNorm1d(d_out)) if bn else None
	block.append(nn.Dropout(p_dropout)) if dropout else None
	block.append(activations)
	
	return nn.Sequential(*block)
	

class SimpleNet(nn.Module):
	# class for neural network with one target variable 
	def __init__(self, d_in, d_hidden, n_layer, d_out, *args, **kwargs):
		super().__init__()
		
		self.enc_sizes= [d_in, * [d_hidden]*n_layer, d_out] # list with sizes of neural networks layers
		self.lin_blocks = nn.Sequential(*[LinearBlock(in_l, out_l, *args, **kwargs) for in_l, out_l in zip(self.enc_sizes, self.enc_sizes[1:])])

	def forward(self, x):
		return self.lin_blocks(x)


class ResidualBlock(nn.Module):
	# class for residual blocks, these are the basic building blocks of ResNet
	# Skip connections between layers facilitate training of deeper networks
	# Input is propagated forward and added unchanged to the output of the Residual Block
	def __init__(self, d_hidden, n_skip=2, bn=True, dropout=True, activation='elu', p_dropout=.1):
		super().__init__()
		
		self.block = []
		for i in range(n_skip):
			self.block.append(nn.Linear(d_hidden, d_hidden))
			self.block.append(nn.BatchNorm1d(d_hidden)) if bn else None
			self.block.append(nn.Dropout(p_dropout)) if dropout else None
			self.block.append(activation_func(activation))
		
		self.block = nn.Sequential(*self.block)
	
	def forward(self, x):
		return x + self.block(x)


class SimpleResNet(nn.Module):
	# class for neural networks with residual blocks and one target variable
	def __init__(self, d_in, d_hidden, n_layer, d_out, n_skip=2, *args, **kwargs):
		super().__init__()
		
		self.blocks_res = [LinearBlock(d_in=d_in,d_out=d_hidden, *args, **kwargs)]

		for i in range(int(n_layer/n_skip)):
			self.blocks_res.append(ResidualBlock(d_hidden=d_hidden, n_skip=n_skip,  *args, **kwargs))
		
		for i in range(n_layer%n_skip):
			self.blocks_res.append(LinearBlock(d_in=d_hidden,d_out=d_hidden, *args, **kwargs))
		
		self.blocks_res.append(LinearBlock(d_in=d_hidden,d_out=d_out, *args, **kwargs))
		self.blocks_res=nn.Sequential(*self.blocks_res)
	
	def forward(self, x):
		return self.blocks_res(x)

class Net(nn.Module):
	# class for neural network with two target variables. One feature extractor network and two separate prediction networks and the gradient reversal layer
	def __init__(self, n_cov, d_hidden = 5, n_layer_feat = 1, n_layer_head = 1, *args, **kwargs):
		super().__init__()
		
		self.feature_extractor = SimpleNet(d_in=n_cov, d_hidden=d_hidden, n_layer=n_layer_feat, d_out=d_hidden, *args, **kwargs)

		self.predictor_owl = SimpleNet(d_in=d_hidden, d_hidden=d_hidden, n_layer=n_layer_head, d_out=1, *args, **kwargs)
		self.predictor_adv = SimpleNet(d_in=d_hidden, d_hidden=d_hidden, n_layer=n_layer_head, d_out=1, *args, **kwargs)

	def forward(self, x, grl_lambda=1):
		
		features = self.feature_extractor(x)
		pred_owl = self.predictor_owl(features) # no gradient reversal on owl prediction network

		features_grl = GradientReversalFn.apply(features, grl_lambda) # We apply the gradient reversal layer before propagating the input to the second prediction network
		pred_adv = self.predictor_adv(features_grl) 
		
		return pred_owl, pred_adv
 

class ResNet(nn.Module):
	# class for neural network with two target variables and gradient reversal
	# one feature extractor network and two separate prediction networks and the gradient reversal layer
	def __init__(self, n_cov, d_hidden = 5, n_layer_feat = 1, n_layer_head = 1, *args, **kwargs):
		super().__init__()
		
		self.feature_extractor = SimpleResNet(d_in=n_cov, d_hidden=d_hidden, n_layer=n_layer_feat, d_out=d_hidden, *args, **kwargs)

		self.predictor_owl = SimpleResNet(d_in=d_hidden, d_hidden=d_hidden, n_layer=n_layer_head, d_out=1, *args, **kwargs)
		self.predictor_adv = SimpleResNet(d_in=d_hidden, d_hidden=d_hidden, n_layer=n_layer_head, d_out=1, *args, **kwargs)

	def forward(self, x, grl_lambda=1):
		
		features = self.feature_extractor(x)
		pred_owl = self.predictor_owl(features) # no gradient reversal on owl prediction network

		features_grl = GradientReversalFn.apply(features, grl_lambda) # We apply the gradient reversal layer before propagating the input to the second prediction network
		pred_adv = self.predictor_adv(features_grl)
		
		return pred_owl, pred_adv

class GradientReversalFn(Function):
	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha
		return x        # No Operation in the forward pass. Simply propagating x further
		
	@staticmethod
	def backward(ctx, grad_output):
		# grad_output: gradient so far as calculated by backprop, dL/dx
		output = - ctx.alpha * grad_output # Flipping the gradient by multiplying by alpha
		return output, None
