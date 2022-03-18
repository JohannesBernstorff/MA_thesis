# Codebase-Explanation

## Contents:
- Setup Environment
- Directory Overview
- Run Experiments
- Set Arguments
- Inspect training with tensorboard

## Setup Environment
We choose conda for the dependency management. To install conda we refer to https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html. To create an environment for the project, install dependencies, and activate the virtual environment:

```bash
conda env create -f Code/requirements.yml
conda activate test_env
```
We use DeepCDE to estimate conditional densities. DeepCDE is publicly available on Github and must be cloned in the Code/Data folder of our project.
```bash
cd Code/Data
git clone https://github.com/tpospisi/DeepCDE.git
```

## Directory Overview
As main directory we suggest the subfolder, Code. Below is an overview (with short explanations) of the structure of the folder Code:
```bash
└── Code 
	# folder with the baseline models
	├── Baselines
		├── Chen		# folder with code to replicate Chen's method
		├── indirect		# folder with code to build simple indirect methods
		└── Mi			# folder with code to replicate Mi's method

	# folder with the data simulation scripts
	├── Data
		├── DeepCDE			# where to save the DeepCDE repo
		├── cde_own.py			# script to estimate conditional densities
		├── simulation_settings.py	# script to simulate the different settings
		└── simulation_utils.py		# script with helper functions for the data simulation
	
	# folder with our proposed model architecture and training procedure
	├── Model		
		├── ensemble_fn.py	# script for ensemble learning with our proposed method
		├── loss.py		# script for our proposed loss functions
		├── net.py		# script for our proposed model architectures
		└── train_fn.py		# script for the training procedure

	├── model_params	# folder to save the models after training for ensemble predictions
	├── results		# folder to save the results, csv-docs with the value of IDRs
	├── runs		# folder to save the tf-files for tensorboard
	├── utils		# folder with hepler functions...
	├── z_archive		# folder with old scripts

	# sh-scripts to execute the experiments
	├── all_1_1.sh
	... 			
	├── all_3_1.sh		
	├── flex.sh		
	├── main.py		# main python script that calls and executes the other scripts
	└── requirements.yml	# yaml-file to build virtual environment
```



## Run Experiments
Several experiments were performed to showcase the functionality of our proposed methods. In the experiments section of the MA-thesis, we describe the results on three settings with simple simulation mechanisms, on one setting with confounder simulations and on one setting with complex simulations. We choose different configurations for one Simple Setting and the Confounder Setting to analyse the influence of interaction and discontinuous effects:

1. Simple Setting
	1. Simple Setting 1 - (*setting1_1*)
		1. Simple Setting 1a (*setting1_1a*)
	2. Simple Setting 2 (*setting1_2*)
	3. Simple Setting 3 (*setting1_3*)
2. Confounder Setting (*setting2_1*)
	1. Confounder Setting 1a (*setting2_1a*)
3. Complex Setting (*setting3_1*)

Information and intuition about the simulation settings and configurations can be found in the experiments section of the thesis. The implementation of the simulations can be found in the subfolder Code/Data. The python-scripts simulation_settings.py and simulation_utils.py build the simulated datasets. In simulation_settings.py each simulation setting (e.g. *setting1_1*) is implemented as a class which inherits functionalities and propoerties from the base-class, (*own_sim_base*). We execute sh-scripts to run our proposed methods with the desired parameter configurations, e.g.: 
```bash
cd Code
./flex.sh
```
We can specifiy desired parameters in *Code/flex.sh* and test our proposed methods functionality under different configurations. The results reported in the MA-thesis were obtained with:

```bash
cd Code
./all_1_1.sh # Table 1: Simple Setting 1 with Linear Effects
./all_1_1a.sh # Table 2: Simple Setting 1 with Interaction/Discontinuous Effects
./all_1_2.sh # Table 3: Simple Setting 2 with Linear Effects
./all_1_3.sh # Table 4: Simple Setting 3 with Linear Effects
./all_2_1.sh # Table 5: Under- vs. Over-dosing in the Confounder Setting
./all_2_1a.sh # Table 6-8: Interaction in the Confounder Setting
./all_3_1.sh # Table 9-10: Results in the Complex Setting
```

- **Table 2:  Simple Setting 1 with Interaction/Discontinuous Effects** In *all_1_1a.sh* we specify *gamma_factor*/*delta_factor* = 0.1 for the weak interaction setting and *gamma_factor*/*delta_factor* = 0.5 for the strong interaction setting.
- **Table 5:  Under- vs. Over-dosing in the Confounder Setting.** In *all_2_1.sh* we specify *beta_1* = 0.5 for the overdosing and *beta_1* = 1.5 for the underdosing.
- **Tables 6-10** We need to specify which type of confounder adjustment to use in the sh-scripts *all_2_1a.sh* or *all_3_1.sh*:

Conditional density estimation  | True conditional density | No density weighting | Adversarial Randomization
| :--- | :--- | :--- | :---
*--calculate_con_dens*  | *--no-calculate_con_dens* | *--no-calculate_con_dens* | *--no-calculate_con_dens* 
*--no-with_true_t_dens*  | *--with_true_t_dens* | *--no-with_true_t_dens* | *--no-with_true_t_dens*
*--no-AdvRand* | *--no-AdvRand* | *--no-AdvRand* | *--AdvRand*

- Decrease --n_repeats and --n_ensemble if only want to test results


## Set Arguments:
To train our proposed methods and the baseline methods, the sh-scripts execute the python-script *main.py*. Several arguments can be specified to adjust the configuration of our proposed method, of the baseline methods, of the simulations and of the training procedure. Below we list all arguments used in the python-script *main.py*. These can be added to the sh-scripts as well as the command line when executing the sh-scripts:

### General 
Argument  | What is this for?
------------- | -------------
*--path* | path to main directory? (Code Folder e.g.:'/Users/jbernstorff/polybox/MA_thesis/test_repo/Code/')
*--verbose*  | verbose terminal output ?
*--no-verbose*  | silent terminal output ?
*--num_gpu*  | number of GPUs ? 
*--write_tensorboard*  | write results to tensorboard? only possible for single runs...
*--no-write_tensorboard*  | do not write results to tensorboard ?
*--type_save*  | name of saved data and results ?


### Simulation parameters: How to simulate the data?:
Argument  | What is this for?
------------- | -------------
*--setting*  | which setting to simulate?
*--n_cov*  | how many covariates?
*--n_train*  | number of training observations?
*--n_val*  | number of val observations?
*--n_test*  | number of testing observations?
*--simulate_new*  | simulate new data?
*--no-simulate_new*  | load old data?
*--seed*   | use what seed?
*--noise*  | noise for setting 1_1 for Y?
*--beta_1* | coefficient for T -> Y
*--beta_2* | coefficient for T -> Y to the power of two
*--alpha_factor*  | baseline effect X -> T in confounder setting
*--delta_factor*  | baseline effect X -> Y in confounder setting
*--gamma_factor*  | interaction effect X and T on Y


### Training parameters: how to train our proposed methods?
Argument  | What is this for?
------------- | -------------
*--n_repeats*  | number of repetitions of training ?
*--n_ensemble*  | number of models in ensemble ?
*--n_epochs*  | number of training epochs ?
*--batch_size* | batch size ?
*--lr*  | learning rate ?
*--early_stopping*  | use early stopping ? 
*--no-early_stopping*  | do not use early stopping ? 
*--early_stopping_patience*  | patience before early stopping ?
*--lr_scheduler*  | use learning rate scheduler ?
*--no-lr_scheduler*  | doe not use learning rate scheduler ?
*--weight_decay*  | what weight decay rate ?
*--dropout*  | use dropout ?
*--no-dropout*  | do not use dropout ?
*--p_dropout*  | dropout probability ?
*--bn*  | use batch normalizaiton ?
*--no-bn*  | doe not use batch normailization ?
*--scale_dt*  | scale inputs before training ?
*--no-scale_dt*  | do not scale inputs before training ?


### Model parameters: how to configure our model?
Argument  | What is this for?
------------- | -------------
*--n_layer_feat*  | number of hidden layers for feature generator ?
*--n_layer_head*  | number of hidden layers for prediction heads ?
*--d_hidden*  | number of hidden units ?
*--activation*  | which activation function to use ?


### Confounding correction parameters: use which scheme?
Argument  | What is this for?
------------- | -------------
*--calculate_con_dens*  | Calculate the conditional density for confounder correction?
*--no-calculate_con_dens*  | Do not calculate the conditional density?
*--with_true_t_dens* | Use the true conditional density for confounder correction?
*--no-with_true_t_dens*  | Do not use the true conditional density for confounder correction?
*--AdvRand*  | Use Adversarial Randomization for confounder correction?
*--no-AdvRand*  | Do not use Adversarial Randomization for confounder correction?
*--grl_weight* | weight of reversed gradients


### Loss parameters: how to configure the loss function?
Argument  | What is this for?
------------- | -------------
*--y_pow*  | increase outcome to the power of x?
*--phi*  | set range considered by the loss function ?
*--max_phi*  | set maximum range for dynamic loss functions ?
*--phase1_end*  | set epoch number when dynamic loss 2 kicks in ?


### Mi Model parameters: how to configure the baseline model?
Argument  | What is this for?
------------- | -------------
*--bins*  | how many bins are used for the categorization of continuous outcome variable ?
*--n_layer_out*  | number of hidden layers in outcome model ?
*--n_layer_t*  | number of hidden layers in treatment predictino model ?
*--d_hidden_out*  | number of hidden units in outcome model ?
*--d_hidden_t*  | number of hidden units in treatment prediction model ?

## Tensorboard
Specify
*--write_tensorboard*

Open tensorboard
```bash
tensorboard --logdir=runs
```
