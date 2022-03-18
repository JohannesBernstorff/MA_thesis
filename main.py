import argparse
import pandas as pd

from Model.ensemble_fn import *

from Baselines.Mi.ensemble_fn import *
from Baselines.Chen.chen_base import *
from Baselines.indirect.xgboost_base import *
from Baselines.indirect.svr_base import *
from Baselines.indirect.lasso_base import *

parser = argparse.ArgumentParser()

# General:
parser.add_argument('--path', default='/Users/jbernstorff/polybox/MA_thesis/Supplementary/Code/', type=str)
parser.add_argument('--num_gpu', default=0, type=int)
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--no-verbose', dest='verbose', action='store_false')
parser.add_argument('--write_tensorboard', dest='write_tensorboard', action='store_true')
parser.add_argument('--no-write_tensorboard', dest='write_tensorboard', action='store_false')
parser.add_argument('--type_save', default='handpicked', type=str)

# Simulation parameters: How to simulate the data?:
parser.add_argument('--setting', default='2_1', type=str)
parser.add_argument('--n_cov', default=3, type=int)
parser.add_argument('--n_train', default=800, type=int)
parser.add_argument('--n_val', default=200, type=int)
parser.add_argument('--n_test', default=2000, type=int)
parser.add_argument('--simulate_new', dest='simulate_new', action='store_true')
parser.add_argument('--no-simulate_new', dest='simulate_new', action='store_false')
parser.add_argument('--seed', default=12101995, type=int)
parser.add_argument('--scale_dt', dest='scale_dt', action='store_true')
parser.add_argument('--no-scale_dt', dest='scale_dt', action='store_false')
parser.add_argument('--noise', default=.5, type=float)
parser.add_argument('--beta_1', default=0.5, type=float)
parser.add_argument('--beta_2', default=0.5, type=float)
parser.add_argument('--alpha_factor', default=-0.2, type=float)
parser.add_argument('--delta_factor', default=-0.5, type=float)
parser.add_argument('--gamma_factor', default=0.5, type=float)

# Training parameters: how to train our proposed methods?
parser.add_argument('--n_repeats', default=10, type=int)
parser.add_argument('--n_ensemble', default=10, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--n_epochs', default=1000, type=int)
parser.add_argument('--early_stopping', dest='early_stopping', action='store_true')
parser.add_argument('--no-early_stopping', dest='early_stopping', action='store_false')
parser.add_argument('--early_stopping_patience', default=40, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--lr_scheduler', dest='lr_scheduler', action='store_true')
parser.add_argument('--no-lr_scheduler', dest='lr_scheduler', action='store_false')
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--dropout', dest='dropout', action='store_true')
parser.add_argument('--no-dropout', dest='dropout', action='store_false')
parser.add_argument('--p_dropout', default=.3, type=float)
parser.add_argument('--bn', dest='bn', action='store_true')
parser.add_argument('--no-bn', dest='bn', action='store_false')

# Model parameters: how to configure our model?
parser.add_argument('--n_layer_feat', default=2, type=int)
parser.add_argument('--n_layer_head', default=2, type=int)
parser.add_argument('--d_hidden', default=30, type=int)
parser.add_argument('--activation', default='elu', type=str)

# Confounding correction parameters: use which scheme?
parser.add_argument('--calculate_con_dens', dest='calculate_con_dens', action='store_true')
parser.add_argument('--no-calculate_con_dens', dest='calculate_con_dens', action='store_false')
parser.add_argument('--with_true_t_dens', dest='with_true_t_dens', action='store_true')
parser.add_argument('--no-with_true_t_dens', dest='with_true_t_dens', action='store_false')
parser.add_argument('--AdvRand', dest='AdvRand', action='store_true')
parser.add_argument('--no-AdvRand', dest='AdvRand', action='store_false')
parser.add_argument('--grl_weight', default=.5, type=float)

# Loss parameters: how to configure the loss function?
parser.add_argument('--y_pow', default=1, type=int)
parser.add_argument('--phi', default=1, type=float)
parser.add_argument('--max_phi', default=5, type=float)
parser.add_argument('--phase1_end', default=20, type=float)

# Mi Model parameters: how to configure the baseline model?
parser.add_argument('--bins', default=7, type=int)
parser.add_argument('--n_layer_out', default=2, type=int)
parser.add_argument('--n_layer_t', default=2, type=int)
parser.add_argument('--d_hidden_out', default=30, type=int)
parser.add_argument('--d_hidden_t', default=30, type=int)

args=parser.parse_args()

# Which loss functions to use
loss_fns = [ "my_loss_detached_I", "my_loss_surrogate_I", "my_loss_abs_dev"]
#loss_fns = [ "my_loss_detached_I", "my_loss_surrogate_I", "my_loss_abs_dev", "my_loss_detached_I_dyn1",  "my_loss_surrogate_I_dyn1", "my_loss_detached_I_dyn2", "my_loss_surrogate_I_dyn2"]

all_res = pd.DataFrame(data=None, columns=['mean_test', 'std_test', 'mean_train','std_train']) # dataframe for results

for i, l in enumerate(loss_fns): # loop over the loss functions
	res = fit_repeated(criterion=l, args=args) # fit repeated ensemble
	all_res.loc[l,list(res.keys())] = list(res.values())

all_res.loc['mi',list(res.keys())] = list(res.values())	
res = fit_mi_repeat(args) # fit repeated ensemble of Mi-baseline

res = fit_chen_repeat(args) # fit repeated Chen-baseline
all_res.loc['chen',list(res.keys())] = list(res.values())

#res = fit_xgb_repeat(args) # missing memory on local computer: some segmentation fault!
#all_res.loc['xgboost',list(res.keys())] = list(res.values())

res = fit_svr_repeat(args) # fit repeated of SVR-baseline
all_res.loc['svr',list(res.keys())] = list(res.values())

res = fit_lasso_repeat(args) # fit repeated of Lasso-baseline
all_res.loc['lasso',list(res.keys())] = list(res.values())

'''
r = np.empty((10,2))
r_opt = np.empty((10,2))
for i in range(10):
	dt = which_setting(args)
	r[i,0], r[i,1] = dt.y_obs_mean_train, dt.y_obs_mean_test
	r_opt[i,0], r_opt[i,1] = dt.y_opt_mean_train, dt.y_opt_mean_test
all_res.loc['obs','mean_test'] = np.round(np.mean(r[:,1]),4)
all_res.loc['obs','std_test'] = np.round(np.std(r[:,1]),2)
all_res.loc['obs','mean_train'] = np.round(np.mean(r[:,0]),4)
all_res.loc['obs','std_train'] = np.round(np.std(r[:,0]),2)

all_res.loc['opt','mean_test'] = np.round(np.mean(r_opt[:,1]),4)
all_res.loc['opt','std_test'] = np.round(np.std(r_opt[:,1]),2)
all_res.loc['opt','mean_train'] = np.round(np.mean(r_opt[:,0]),4)
all_res.loc['opt','std_train'] = np.round(np.std(r_opt[:,0]),2)
'''

print(all_res)

path_results=path_model = args.path + f'/Results/{args.setting}_size_{args.n_train+args.n_val}_cal_dens_{args.calculate_con_dens}_with_true{args.with_true_t_dens}.csv'
#all_res.to_csv(path_or_buf=path_results) # save results to csv 
