#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:36:44 2024

@author: forootani
"""

import numpy as np
import torch
import sys
import os
import scipy.io as sio

#print(os.path.dirname(os.path.abspath("")))
#sys.path.append(os.path.dirname(os.path.abspath("")))

cwd = os.getcwd()
#sys.path.append(cwd + '/my_directory')
sys.path.append(cwd)

import matplotlib.pyplot as plt

# General imports
import numpy as np
import torch

# DeePyMoD imports
from deepymod import DeepMoD
from deepymod.data import Dataset, get_train_test_loader
from deepymod.data.samples import Subsample_random
from deepymod.data.burgers import burgers_delta
from deepymod.model.constraint import LeastSquares, Ridge, STRidgeCons
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D
from deepymod.model.sparse_estimators import Threshold, STRidge
from deepymod.training import train
#from deepymod.training.training_2 import train
from deepymod.training.sparsity_scheduler import Periodic, TrainTest, TrainTestPeriodic
#from deepymod.data.data_set_preparation import DatasetPDE, pde_data_loader
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
from deepymod.data.DEIM_class import DEIM
import shutil

from deepymod.utils.utilities import create_or_reset_directory


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


def create_data_1():
    data = loadmat("data/kdv.mat")
    data = scipy.io.loadmat("data/kdv.mat")
    #data = np.load("data/kdv.npy",allow_pickle=True)
    ## preparing and normalizing the input and output data
    t_o = 1 * data["t"].flatten()[0:201, None]
    x_o = 1 * data["x"].flatten()[:, None]
    Exact = np.real(data["usol"])
    deim_instance = DEIM(Exact, 2, t_o.squeeze(), x_o.squeeze(),
                         tolerance = 1 * 1e-5,num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
        
    return coords, data


x, u =  create_data_1()


num_of_samples = 900

dataset_1 = Dataset(
    create_data_1,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)





train_dataloader_1, test_dataloader_1 = get_train_test_loader(
    dataset_1, train_test_split=0.99)


"""
tensor([[ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [-0.9965],
        [ 0.0000],
        [-5.9911],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000]])
"""


##########################
##########################

poly_order = 2
diff_order = 3

n_combinations = (poly_order+1)*(diff_order+1) 
n_features = 1


network_1 = NN(2, [32, 32, 32, 32], 1)

#network = NN(2, [30, 30, 30, 30], 1)

library_1 = Library1D(poly_order, diff_order)
estimator_1 = STRidge(tol=0.4)
sparsity_scheduler_1 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
sparsity_scheduler_1 = Periodic(periodicity=50, initial_iteration=1000)

constraint_1 = STRidgeCons(tol=0.1)

model_1 = DeepMoD(network_1, library_1, estimator_1, constraint_1).to(device)

# Defining optimizer
optimizer_1 = torch.optim.Adam(model_1.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 


foldername_1 = "./data/deepymod/KDV/STR_STR"
#shutil.rmtree(foldername)

create_or_reset_directory(foldername_1)


train(
    model_1,
    train_dataloader_1,
    test_dataloader_1,
    optimizer_1,
    sparsity_scheduler_1,
    log_dir= foldername_1,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=1000,)


###########################################################
###########################################################


def create_data_2():
    data = loadmat("data/kdv.mat")
    data = scipy.io.loadmat("data/kdv.mat")
    #data = np.load("data/kdv.npy",allow_pickle=True)
    ## preparing and normalizing the input and output data
    t_o = 1 * data["t"].flatten()[0:201, None]
    x_o = 1 * data["x"].flatten()[:, None]
    Exact = np.real(data["usol"])
    deim_instance = DEIM(Exact, 2, t_o.squeeze(), x_o.squeeze(),
                         tolerance = 5.001 * 1e-5,num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
        
    return coords, data


x, u =  create_data_2()



num_of_samples = 900

dataset_2 = Dataset(
    create_data_2,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)


train_dataloader_2, test_dataloader_2 = get_train_test_loader(
    dataset_2, train_test_split=0.99)

network_2 = NN(2, [32, 32, 32, 32], 1)

library_2 = Library1D(poly_order, diff_order)
estimator_2 = STRidge(tol=0.4) 
sparsity_scheduler_2 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
sparsity_scheduler_2 = Periodic(periodicity=50, initial_iteration=1000)

constraint_2 = STRidgeCons(tol=0.1)

model_2 = DeepMoD(network_2, library_2, estimator_2, constraint_2).to(device)

# Defining optimizer
optimizer_2 = torch.optim.Adam(model_2.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername_2 = "./data/deepymod/KDV/STR_ols"
#shutil.rmtree(foldername)

create_or_reset_directory(foldername_2)

train(model_2,
    train_dataloader_2,
    test_dataloader_2,
    optimizer_2,
    sparsity_scheduler_2,
    log_dir= foldername_2,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=1000,)


###########################################################
###########################################################

def create_data_3():
    data = loadmat("data/kdv.mat")
    data = scipy.io.loadmat("data/kdv.mat")
    #data = np.load("data/kdv.npy",allow_pickle=True)
    ## preparing and normalizing the input and output data
    t_o = 1 * data["t"].flatten()[0:201, None]
    x_o = 1 * data["x"].flatten()[:, None]
    Exact = np.real(data["usol"])
    deim_instance = DEIM(Exact, 2, t_o.squeeze(), x_o.squeeze(),
                         tolerance = 1 * 1e-6,num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
        
    return coords, data

x, u =  create_data_3()


num_of_samples = 900

dataset_3 = Dataset(
    create_data_3,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)


train_dataloader_3, test_dataloader_3 = get_train_test_loader(
    dataset_3, train_test_split=0.99)


network_3 = NN(2, [32, 32, 32, 32], 1)

library_3 = Library1D(poly_order, diff_order)
estimator_3 = STRidge(tol=0.4) 
sparsity_scheduler_3 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
sparsity_scheduler_3 = Periodic(periodicity=50, initial_iteration=1000)

constraint_3 = STRidgeCons(tol=0.1)

model_3 = DeepMoD(network_3, library_3, estimator_3, constraint_3).to(device)

# Defining optimizer
optimizer_3 = torch.optim.Adam(model_3.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername_3 = "./data/deepymod/KDV/STR_ols"
#shutil.rmtree(foldername)

create_or_reset_directory(foldername_3)

train(model_3,
    train_dataloader_3,
    test_dataloader_3,
    optimizer_3,
    sparsity_scheduler_3,
    log_dir= foldername_3,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=1000,)



########################################################
########################################################



def create_data_4():
    data = loadmat("data/kdv.mat")
    
    data = scipy.io.loadmat("data/kdv.mat")
    #data = np.load("data/kdv.npy",allow_pickle=True)

    ## preparing and normalizing the input and output data
    t_o = 1 * data["t"].flatten()[0:201, None]
    x_o = 1 * data["x"].flatten()[:, None]
    Exact = np.real(data["usol"])
    #X, T = np.meshgrid(x, t, indexing="ij")
    #X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    #u_star = Exact.flatten()[:, None]
    
    
    deim_instance = DEIM(Exact, 2, t_o.squeeze(), x_o.squeeze(),
                         tolerance = 1 * 1e-7, num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
        
    return coords, data



x, u =  create_data_4()





num_of_samples = 900


dataset_4 = Dataset(
    create_data_4,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)


train_dataloader_4, test_dataloader_4 = get_train_test_loader(
    dataset_4, train_test_split=0.99)


##########################
##########################

poly_order = 2
diff_order = 3

n_combinations = (poly_order+1)*(diff_order+1) 
n_features = 1


network_4 = NN(2, [32, 32, 32, 32], 1)

#network = NN(2, [30, 30, 30, 30], 1)

library_4 = Library1D(poly_order, diff_order)

sparsity_scheduler_4 = Periodic(periodicity=50, initial_iteration=1000)

constraint_4 = STRidgeCons(tol = 0.1)

estimator_4 = STRidge(lam = 0.0000, maxit = 100, tol = 0.4)

model_4 = DeepMoD(network_4, library_4, estimator_4, constraint_4).to(device)

# Defining optimizer
optimizer_4 = torch.optim.Adam(model_4.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 


foldername_4 = "./data/deepymod/kdv/tol_deim_sens_7"
create_or_reset_directory(foldername_4)

train(
    model_4,
    train_dataloader_4,
    test_dataloader_4,
    optimizer_4,
    sparsity_scheduler_4,
    log_dir= foldername_4,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    #patience=1000,
    )





########################################################
########################################################



print(model_1.constraint.coeff_vectors[0].detach().cpu())
print(model_2.constraint.coeff_vectors[0].detach().cpu())
print(model_3.constraint.coeff_vectors[0].detach().cpu())
print(model_4.constraint.coeff_vectors[0].detach().cpu())











