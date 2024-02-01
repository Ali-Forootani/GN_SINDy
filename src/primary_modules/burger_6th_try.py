#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 08:27:24 2024

@author: forootani
"""

import numpy as np
import torch
import sys
import os
import scipy.io as sio
cwd = os.getcwd()
sys.path.append(cwd)

import matplotlib.pyplot as plt
# General imports
import numpy as np
import torch

# DeePyMoD imports
from deepymod import DeepMoD
from deepymod.data import Dataset, get_train_test_loader
from deepymod.data.samples import Subsample_random
from deepymod.data.burgers import burgers_delta, burgers_delta_org
from deepymod.model.constraint import LeastSquares, Ridge, STRidgeCons
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D
from deepymod.model.sparse_estimators import Threshold, STRidge
from deepymod.training import train
from deepymod.data.DEIM_class import DEIM
from deepymod.analysis import load_tensorboard
import shutil

from deepymod.training.sparsity_scheduler import Periodic, TrainTest, TrainTestPeriodic

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
v = 0.1
A = 1.0

x = torch.linspace(-4, 4, 500)
t = torch.linspace(0.5, 10.0, 500)

load_kwargs = {"x": x, "t": t, "v": v, "A": A}
preprocess_kwargs = {"noise_level": 0.00}

#######################
#######################
import scipy.io

def create_data():
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

coords_2, data_2 = create_data()


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


model = DeepMoD(network, library, estimator, constraint, estimator).to(device)
# Defining optimizer
optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername = "./data/deepymod/burgers/STR_STR"
#shutil.rmtree(foldername)

#######################################
#######################################
#######################################

from deepymod.utils.utilities import create_or_reset_directory

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

foldername_2 = "./data/deepymod/burgers/Lasso_OLS"
#shutil.rmtree(foldername)

#######################################
#######################################
#######################################




create_or_reset_directory(foldername_2)

network_2 = NN(2, [64, 64, 64, 64], 1)

estimator_2 = Threshold(0.05)
 
sparsity_scheduler_2 = TrainTestPeriodic(periodicity=100, patience=500, delta=1e-5)
sparsity_scheduler_2 = Periodic(periodicity=100, initial_iteration=500)



constraint_2 = LeastSquares()

library_2 = Library1D(poly_order, diff_order)

model_2 = DeepMoD(network_2, library_2, estimator_2, constraint_2, estimator).to(device)

optimizer_2 = torch.optim.Adam(model_2.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3)


train( model_2, train_dataloader, test_dataloader, optimizer_2, sparsity_scheduler_2,
    log_dir= foldername_2, exp_ID="Test", write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=200,
)

model_2.sparsity_masks

print(model_2.constraint.coeff_vectors[0].detach().cpu())



###########################################################################
###########################################################################
###########################################################################
################## Third setup for the simulations --- STR-OLS

foldername_3 = "./data/deepymod/burgers/STR_OLS"
#shutil.rmtree(foldername)

#######################################
#######################################
#######################################





create_or_reset_directory(foldername_3)


network_3 = NN(2, [64, 64, 64, 64], 1)
sparsity_scheduler_3 = TrainTestPeriodic(periodicity=100, patience=500, delta=1e-5)
sparsity_scheduler_3 = Periodic(periodicity=100, initial_iteration=500)

estimator_3 = STRidge() 
constraint_3 = LeastSquares()

library_3 = Library1D(poly_order, diff_order)
model_3 = DeepMoD(network_3, library_3, estimator_3, constraint_3, estimator).to(device)
optimizer_3 = torch.optim.Adam(model_3.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3)


train( model_3, train_dataloader, test_dataloader, optimizer_3, sparsity_scheduler_3,
    log_dir= foldername_3, exp_ID="Test", write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=200,)

model_3.sparsity_masks

print(model_3.constraint.coeff_vectors[0].detach().cpu())


###########################################################################
###########################################################################
###########################################################################
################## Fourth setup for the simulations --- LASSO-STR

foldername_4 = "./data/deepymod/burgers/Lasso_STR"
#shutil.rmtree(foldername)

#######################################
#######################################
#######################################



create_or_reset_directory(foldername_4)


network_4 = NN(2, [64, 64, 64, 64], 1)
sparsity_scheduler_4 = TrainTestPeriodic(periodicity=100, patience=500, delta=1e-5)
sparsity_scheduler_4 = Periodic(periodicity=100, initial_iteration=500)


estimator_4 = Threshold(0.05) 
constraint_4 = STRidgeCons()

library_4 = Library1D(poly_order, diff_order)
model_4 = DeepMoD(network_4, library_4, estimator_4, constraint_4, estimator).to(device)
optimizer_4 = torch.optim.Adam(model_4.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3)


train( model_4, train_dataloader, test_dataloader, optimizer_4, sparsity_scheduler_4,
    log_dir= foldername_4, exp_ID="Test", write_iterations=25,
    max_iterations=25000,
    #delta=1e-4,
    #patience=1000,
)

model_4.sparsity_masks

print(model_4.constraint.coeff_vectors[0].detach().cpu())




