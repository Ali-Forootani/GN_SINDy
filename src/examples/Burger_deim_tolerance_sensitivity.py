#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:39:31 2024

@author: forootani
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Burger DEIM algorithm to analyse its performance under different 

precision value for Q-DEIM algorithm

"""

import numpy as np
import torch
import sys
import os
import scipy.io as sio

cwd = os.getcwd()
#sys.path.append(cwd + '/my_directory')
sys.path.append(cwd)


def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir
root_dir = setting_directory(2)


import matplotlib.pyplot as plt

# General imports
import numpy as np
import torch

# DeePyMoD imports
from GNSINDy.src.deepymod import DeepMoD
from GNSINDy.src.deepymod.data import Dataset, get_train_test_loader
from GNSINDy.src.deepymod.data.samples import Subsample_random
from GNSINDy.src.deepymod.data.burgers import burgers_delta, burgers_delta_org
from GNSINDy.src.deepymod.data.burgers import burgers_delta
from GNSINDy.src.deepymod.model.constraint import LeastSquares, Ridge, STRidgeCons
from GNSINDy.src.deepymod.model.func_approx import NN
from GNSINDy.src.deepymod.model.library import Library1D
from GNSINDy.src.deepymod.model.sparse_estimators import Threshold, STRidge
from GNSINDy.src.deepymod.training import train
#from deepymod.training.training_2 import train
from GNSINDy.src.deepymod.training.sparsity_scheduler import Periodic, TrainTest, TrainTestPeriodic
#from deepymod.data.data_set_preparation import DatasetPDE, pde_data_loader
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
from GNSINDy.src.deepymod.data.DEIM_class import DEIM
import shutil

from GNSINDy.src.deepymod.utils.utilities import create_or_reset_directory

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(50)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


###############################################
###############################################

# Making dataset


#######################
#######################
import scipy.io

def create_data():
    
    x_o = torch.linspace(-8, 8, 100)
    t_o = torch.linspace(0.5, 10.0, 100)
    v = 0.1
    A = 1.0    
    _ , Exact = burgers_delta_org( x_o, t_o, v, A)
    
    deim_instance = DEIM(Exact, 2, t_o, x_o, tolerance = 1e-3, num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
    
    return coords, data

coords, data = create_data()


###########################
###########################


num_of_samples = 50


#########################
#########################
#########################


dataset = Dataset(
    create_data,
    #load_kwargs=load_kwargs,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)


train_dataloader, test_dataloader = get_train_test_loader(
    dataset, train_test_split = 1.00)

##########################
##########################

poly_order = 2
diff_order = 2


network = NN(2, [64, 64, 64, 64], 1)
library = Library1D(poly_order, diff_order)
sparsity_scheduler = TrainTestPeriodic(periodicity=100, patience=500, delta=1e-5)


#constraint = Ridge()
constraint = STRidgeCons()
estimator = STRidge()


model = DeepMoD(network, library, estimator, constraint).to(device)
# Defining optimizer
optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername = "./data/deepymod/burgers/tolerance_3"
#shutil.rmtree(foldername)

#######################################
#######################################
#######################################


create_or_reset_directory(foldername)


########################################
########################################
########################################
############## First setup for the simulatons --- STR-STR


train( model, train_dataloader, test_dataloader, optimizer, sparsity_scheduler,
    log_dir= foldername, exp_ID="Test", write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=200,
)

model.sparsity_masks

print(model.constraint.coeff_vectors[0].detach().cpu())

###########################################################################
###########################################################################
###########################################################################
################## Second setup for the simulations --- Lasso-OLS

foldername_2 = "./data/deepymod/burgers/tolerance_4"
#shutil.rmtree(foldername)

#######################################
#######################################
#######################################



def create_data_2():
    #data = scipy.io.loadmat("deepymod/data/numerical_data/burgers.mat")
    
    x_o = torch.linspace(-8, 8, 100)
    t_o = torch.linspace(0.5, 10.0, 100)
    v = 0.1
    A = 1.0    
    _ , Exact = burgers_delta_org( x_o, t_o, v, A)
    
    deim_instance = DEIM(Exact, 2, t_o, x_o, tolerance = 1e-4, num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
    
    return coords, data

coords_2, data_2 = create_data_2()



dataset_2 = Dataset(
    create_data_2,
    #load_kwargs=load_kwargs,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)



train_dataloader_2, test_dataloader_2 = get_train_test_loader(dataset_2, train_test_split = 1.00)



create_or_reset_directory(foldername_2)

network_2 = NN(2, [64, 64, 64, 64], 1)

estimator_2 = STRidge()
 
sparsity_scheduler_2 = Periodic(periodicity=100, initial_iteration=500)


constraint_2 = STRidgeCons()
library_2 = Library1D(poly_order, diff_order)
model_2 = DeepMoD(network_2, library_2, estimator_2, constraint_2).to(device)
optimizer_2 = torch.optim.Adam(model_2.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3)
train( model_2, train_dataloader_2, test_dataloader_2, optimizer_2, sparsity_scheduler_2,
    log_dir= foldername_2, exp_ID="Test", write_iterations=25, max_iterations=25000,
    delta=1e-4, patience=200,)

model_2.sparsity_masks

print(model_2.constraint.coeff_vectors[0].detach().cpu())



###########################################################################
###########################################################################
###########################################################################

foldername_3 = "./data/deepymod/burgers/tolerance_5"
#shutil.rmtree(foldername)

#######################################
#######################################
#######################################



def create_data_3():
    #data = scipy.io.loadmat("deepymod/data/numerical_data/burgers.mat")
    
    x_o = torch.linspace(-8, 8, 100)
    t_o = torch.linspace(0.5, 10.0, 100)
    v = 0.1
    A = 1.0    
    _ , Exact = burgers_delta_org( x_o, t_o, v, A)
    
    deim_instance = DEIM(Exact, 2, t_o, x_o, tolerance = 1e-5, num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
    
    return coords, data

coords_3, data_3 = create_data_3()




dataset_3 = Dataset(
    create_data_3,
    #load_kwargs=load_kwargs,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)

train_dataloader_3, test_dataloader_3 = get_train_test_loader(dataset_3, train_test_split = 1.00)


create_or_reset_directory(foldername_3)

network_3 = NN(2, [64, 64, 64, 64], 1)
sparsity_scheduler_3 = TrainTestPeriodic(periodicity=100, patience=500, delta=1e-5)
sparsity_scheduler_3 = Periodic(periodicity=100, initial_iteration=500)



estimator_3 = STRidge() 
constraint_3 = STRidgeCons()

library_3 = Library1D(poly_order, diff_order)
model_3 = DeepMoD(network_3, library_3, estimator_3, constraint_3).to(device)
optimizer_3 = torch.optim.Adam(model_3.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3)


train( model_3, train_dataloader_3, test_dataloader_3, optimizer_3, sparsity_scheduler_3,
    log_dir= foldername_3, exp_ID="Test", write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=200,)

model_3.sparsity_masks

print(model_3.constraint.coeff_vectors[0].detach().cpu())


###########################################################################
###########################################################################
###########################################################################


foldername_4 = "./data/deepymod/burgers/tolerance_6"
#shutil.rmtree(foldername)

#######################################
#######################################
#######################################



def create_data_4():
    #data = scipy.io.loadmat("deepymod/data/numerical_data/burgers.mat")
    
    x_o = torch.linspace(-8, 8, 100)
    t_o = torch.linspace(0.5, 10.0, 100)
    v = 0.1
    A = 1.0    
    _ , Exact = burgers_delta_org( x_o, t_o, v, A)
    
    deim_instance = DEIM(Exact, 1, t_o, x_o, tolerance = 1e-6, num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
    
    return coords, data


coords_4, data_4 = create_data_4()



dataset_4 = Dataset(
    create_data_4,
    #load_kwargs=load_kwargs,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)

train_dataloader_4, test_dataloader_4 = get_train_test_loader(dataset_4, train_test_split = 1.00)


create_or_reset_directory(foldername_4)

network_4 = NN(2, [64, 64, 64, 64], 1)
sparsity_scheduler_4 = TrainTestPeriodic(periodicity=100, patience=500, delta=1e-5)
sparsity_scheduler_4 = Periodic(periodicity=100, initial_iteration=500)

estimator_4 = STRidge() 
constraint_4 = STRidgeCons()

library_4 = Library1D(poly_order, diff_order)
model_4 = DeepMoD(network_3, library_4, estimator_4, constraint_4).to(device)
optimizer_4 = torch.optim.Adam(model_4.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3)


train( model_4, train_dataloader_4, test_dataloader_4, optimizer_4, sparsity_scheduler_4,
    log_dir= foldername_4, exp_ID="Test", write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=200,)

model_4.sparsity_masks

print(model_4.constraint.coeff_vectors[0].detach().cpu())



