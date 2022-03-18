import time

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from Data.simulation_settings import *

from Model.loss import *
from Model.net import *
from Model.train_fn import *

from utils.general import *

def fit_model(i, criterion, model, path_model, dt, args):
	"""
	function to fit one model
	:params i: number of repetition
	:params criterion: which criterion to use in the OWL objective? pass initialized loss function class
	:params model: pass initiliazed model to the function
	:params path_model: where to save the trained model
	:params dt: pass data class on, which setting, which generating mechanism, which data
	:params args: pass the other arguments on to the training function
	"""
	
	model.load_state_dict(torch.load(path_model+'initial.pth')) # we load the initial weights from model_params

	criterion_owl = initialize_criterion(criterion, args) # initialize the criterion for the OWL objective
	criterion_adv = my_l2() # initialize the criterion for the adversary

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # initialize the optimizer with parameters from args
	
	if args.lr_scheduler: # initialize the learning rate scheduler with parameters from args
		scheduler = ReduceLROnPlateau(
				optimizer, 
				mode='min',
				patience=20, # how many steps/epochs to wait before reducing learning rate 
				cooldown=0, # how many steps/epochs after reduction should wait before again to start counting 
				verbose=False)
	else:
		scheduler = None

	if args.early_stopping:
		early_stopping = EarlyStopping(patience=args.early_stopping_patience) # initialize early stopping from utils/general.py

	# Initialize training class:
	worker=worker_owl(cal_cf_y=dt.generate_value, optim=optimizer, scheduler=scheduler, criterion_owl=criterion_owl, criterion_adv=criterion_adv, AdvRand=args.AdvRand, grl_weight=args.grl_weight)

	# obtain bootstrapped training data
	train_dataloader, val_dataloader = dt.build_dataloader_boot(batch_size=args.batch_size)
	
	# initialize tensorboard
	if args.write_tensorboard:
		log_dir = f'runs/own_sim/{type(dt).__name__}/{args.type_save}_{criterion_owl}'
		writer = SummaryWriter(log_dir=log_dir)
	else:
		writer = None

	# Start training updates:
	for epoch in range(args.n_epochs): 
		# one training step followed by one validation step
		loss_owl_train, loss_adv_train, value_owl_train, value_adv_train = worker.train(model, train_dataloader, dt.device)
		loss_owl_val, loss_adv_val, value_owl_val, value_adv_val = worker.validate(model, val_dataloader, dt.device)
		
		if args.early_stopping: 
			early_stopping(worker.loss_owl/dt.n_val_boot) # check early stopping criterion on 
			if early_stopping.early_stop:
				if args.verbose:
					print(f"INFO: Early stopping at epoch: {epoch}")
				break
		
		if args.write_tensorboard: # write the most important metrics to tensorboard
			writer.add_scalar('loss_owl_train', loss_owl_train/dt.n_train_boot,epoch)
			writer.add_scalar('value_owl_train', value_owl_train/dt.n_train_boot,epoch)
			writer.add_scalar('loss_owl_val', loss_owl_val/dt.n_val_boot, epoch)
			writer.add_scalar('value_owl_val', value_owl_val/dt.n_val_boot,epoch)
			
	if args.write_tensorboard:
		#writer.add_hparams({'lr': args.lr, 'batch_size': args.batch_size, 'wd':args.weight_decay, 'n_layers':args.n_layer_feat+args.n_layer_head, 'd_hidden':args.d_hidden, 'n_train': args.n_train},{'loss_owl_test': loss_owl_test/dt.n_test, ' value_owl_test': value_owl_test/dt.n_test})
		writer.close()

	torch.save(model.state_dict(), path_model+str(i)+'.pth') # save fully trained model
	
	return loss_owl_val/dt.n_val_boot # return validation loss on OOB-samples


def fit_ensemble(criterion, args):
	"""
	function to fit an ensemble
	:params criterion: which criterion to use in the OWL objective? pass initialized loss function class
	:params args: pass the other arguments on to the training function
	"""
	dt = which_setting(args=args) # initialize the simulation data
	if args.verbose:
		print(f"INFO: initializing simulation {type(dt).__name__}!! with loss function: {criterion}")
	
	# initialize a Resnet structure along the parameters passed in args
	model = ResNet(n_cov=dt.n_cov, d_hidden = args.d_hidden, n_layer_feat = args.n_layer_feat, n_layer_head = args.n_layer_head, activation=args.activation, bn=args.bn, dropout=args.dropout, p_dropout=args.p_dropout)

	# send model to GPUs
	if torch.cuda.device_count() > 1 and args.num_gpu > 1:
		model = nn.DataParallel(model, device_ids=[0,1])
	model.to(dt.device)

	path_model = args.path+f'model_params/{type(dt).__name__}/' # where to save the ensemble?
	torch.save(model.state_dict(), path_model+'initial.pth') # save initial, random weights of neural network
	
	results = np.array([fit_model(x, criterion, model, path_model, dt, args) for x in range(args.n_ensemble)]) # fit the separate model and return the validation loss

	sorted_scores = results.argsort() # find best models based on the validation loss
	value_train, value_test=evaluate_ensemble(model, dt=dt, path_model=path_model, sorted_scores=sorted_scores, device=dt.device)
	
	if args.verbose:
		print(f"TRAINING: Value of PREDICTED IDR_owl: {value_train/(dt.n_train+dt.n_val):.3f}")
		print(f"TRAINING: Value of OBSERVED IDR: {dt.y_obs_mean_train:.3f}")
		print(f"TRAINING: Value of OPTIMAL IDR: {dt.y_opt_mean_train:.3f}")
		print('-')
		print(f"TESTING: Value of PREDICTED IDR_owl: {value_test/dt.n_test:.3f}")
		print(f"TESTING: Value of OBSERVED IDR: {dt.y_obs_mean_test:.3f}")
		print(f"TESTING: Value of OPTIMAL IDR: {dt.y_opt_mean_test:.3f}")
		print('-')
		print('-')
	
	return value_train/(dt.n_train+dt.n_val), value_test/dt.n_test 


def fit_repeated(criterion, args):
	"""
	function to repeat the fitting of ensembles
	:params criterion: which criterion to use in the OWL objective? pass initialized loss function class
	:params args: pass the other arguments on to the training function
	"""
	
	if args.verbose:
		start = time.time()
		print(f'Start treatment model fitting: {criterion}, size: {args.n_train+args.n_val}')
	
	results = np.array([fit_ensemble(criterion=criterion, args=args) for x in range(args.n_repeats)])
	
	if args.verbose:
		end = time.time()
		print(f"Training time: {(end-start)/60:.3f} minutes")

	return {'mean_test': np.round(np.mean(results[:,1]), decimals=4),
		'std_test': np.round(np.std(results[:,1]), decimals=2),
		'mean_train': np.round(np.mean(results[:,0]), decimals=4),
		'std_train':  np.round(np.std(results[:,0]), decimals=2)}
