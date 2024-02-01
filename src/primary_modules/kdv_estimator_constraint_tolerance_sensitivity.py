#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:31:02 2024

@author: forootani
"""

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


##########################
##########################

poly_order = 2
diff_order = 3

n_combinations = (poly_order+1)*(diff_order+1) 
n_features = 1


network_1 = NN(2, [32, 32, 32, 32], 1)

#network = NN(2, [30, 30, 30, 30], 1)

library_1 = Library1D(poly_order, diff_order)
estimator_1 = STRidge()
sparsity_scheduler_1 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
constraint_1 = STRidgeCons()

model_1 = DeepMoD(network_1, library_1, estimator_1, constraint_1, estimator_1).to(device)

# Defining optimizer
optimizer_1 = torch.optim.Adam(model_1.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 


foldername_1 = "./data/deepymod/KDV/STR_STR_tol_1_5"
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
                         tolerance = 5 * 1e-5,num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
        
    return coords, data



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

#network = NN(2, [30, 30, 30, 30], 1)

library_2 = Library1D(poly_order, diff_order)
estimator_2 = STRidge() 
sparsity_scheduler_2 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
constraint_2 = STRidgeCons()

model_2 = DeepMoD(network_2, library_2, estimator_2, constraint_2, estimator_2).to(device)

# Defining optimizer
optimizer_2 = torch.optim.Adam(model_2.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 


foldername_2 = "./data/deepymod/KDV/STR_STR_tol_5_5"
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

"""
tensor([[ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [-1.0973],
        [ 0.0000],
        [-5.8382],
        [ 0.0000],
        [ 1.4881],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [-2.7636]])
"""

#######################################################
#######################################################



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
estimator_3 = STRidge() 
sparsity_scheduler_3 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
constraint_3 = STRidgeCons()

model_3 = DeepMoD(network_3, library_3, estimator_3, constraint_3, estimator_3).to(device)

# Defining optimizer
optimizer_3 = torch.optim.Adam(model_3.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername_3 = "./data/deepymod/KDV/STR_STR_tol_6"
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


"""
tensor([[  0.0000],
        [ -0.2812],
        [  0.0000],
        [ -0.7824],
        [  0.0000],
        [ -1.2154],
        [  0.0000],
        [  0.0000],
        [  0.0000],
        [-18.5952],
        [  0.0000],
        [ -7.7209]])
"""

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
    deim_instance = DEIM(Exact, 2, t_o.squeeze(), x_o.squeeze(),
                         tolerance = 5 * 1e-7,num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
        
    return coords, data



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


network_4 = NN(2, [32, 32, 32, 32], 1)


library_4 = Library1D(poly_order, diff_order)
estimator_4 = STRidge()
sparsity_scheduler_4 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
constraint_4 = STRidgeCons()

model_4 = DeepMoD(network_4, library_4, estimator_4, constraint_4, estimator_4).to(device)

# Defining optimizer
optimizer_4 = torch.optim.Adam(model_4.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername_4 = "./data/deepymod/KDV/STR_STR_tol_5_7"
#shutil.rmtree(foldername)

create_or_reset_directory(foldername_4)

train(model_4,
    train_dataloader_4,
    test_dataloader_4,
    optimizer_4,
    sparsity_scheduler_4,
    log_dir= foldername_4,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=1000,)


"""
tensor([[ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [-1.1008],
        [ 0.0000],
        [-5.8001],
        [ 0.0000],
        [ 1.7415],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [-3.2994]])
"""


########################################################
########################################################
########################################################
########################################################
########################################################
########################################################



def create_data_5():
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



num_of_samples = 900

dataset_5 = Dataset(
    create_data_5,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)



train_dataloader_5, test_dataloader_5 = get_train_test_loader(
    dataset_5, train_test_split=0.99)



poly_order = 2
diff_order = 3

n_combinations = (poly_order+1)*(diff_order+1) 
n_features = 1


network_5 = NN(2, [32, 32, 32, 32], 1)

#network = NN(2, [30, 30, 30, 30], 1)

library_5 = Library1D(poly_order, diff_order)
estimator_5 = Threshold(0.1)
sparsity_scheduler_5 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
constraint_5 = STRidgeCons()

model_5 = DeepMoD(network_5, library_5, estimator_5, constraint_5, estimator_5).to(device)

# Defining optimizer
optimizer_5 = torch.optim.Adam(model_5.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 


foldername_5 = "./data/deepymod/KDV/STR_STR_tol_1_5"
#shutil.rmtree(foldername)

create_or_reset_directory(foldername_5)


train(
    model_5,
    train_dataloader_5,
    test_dataloader_5,
    optimizer_5,
    sparsity_scheduler_5,
    log_dir= foldername_5,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=1000,)


"""
tensor([[ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [-0.9911],
        [ 0.0000],
        [-5.9752],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000]])
"""

###########################################################
###########################################################



def create_data_6():
    data = loadmat("data/kdv.mat")
    data = scipy.io.loadmat("data/kdv.mat")
    #data = np.load("data/kdv.npy",allow_pickle=True)
    ## preparing and normalizing the input and output data
    t_o = 1 * data["t"].flatten()[0:201, None]
    x_o = 1 * data["x"].flatten()[:, None]
    Exact = np.real(data["usol"])
    deim_instance = DEIM(Exact, 2, t_o.squeeze(), x_o.squeeze(),
                         tolerance = 5 * 1e-5,num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
        
    return coords, data



num_of_samples = 900

dataset_6 = Dataset(
    create_data_6,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)



train_dataloader_6, test_dataloader_6 = get_train_test_loader(
    dataset_6, train_test_split=0.99)



network_6 = NN(2, [32, 32, 32, 32], 1)

#network = NN(2, [30, 30, 30, 30], 1)

library_6 = Library1D(poly_order, diff_order)
estimator_6 = Threshold(0.1) 
sparsity_scheduler_6 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
constraint_6 = STRidgeCons()

model_6 = DeepMoD(network_6, library_6, estimator_6, constraint_6, estimator_6).to(device)

# Defining optimizer
optimizer_6 = torch.optim.Adam(model_6.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername_6 = "./data/deepymod/KDV/lasso_STR_tol_5_5"
#shutil.rmtree(foldername)

create_or_reset_directory(foldername_6)


train(model_6,
    train_dataloader_6,
    test_dataloader_6,
    optimizer_6,
    sparsity_scheduler_6,
    log_dir= foldername_6,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=1000,)

"""
tensor([[ 0.0000],
        [-0.0305],
        [ 0.0000],
        [-0.9590],
        [ 0.0000],
        [-5.7847],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000]])
"""

#######################################################
#######################################################


def create_data_7():
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



num_of_samples = 900

dataset_7 = Dataset(
    create_data_7,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)


train_dataloader_7, test_dataloader_7 = get_train_test_loader(
    dataset_7, train_test_split=0.99)


network_7 = NN(2, [32, 32, 32, 32], 1)

library_7 = Library1D(poly_order, diff_order)
estimator_7 = Threshold(0.1) 
sparsity_scheduler_7 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
constraint_7 = STRidgeCons()

model_7 = DeepMoD(network_7, library_7, estimator_7, constraint_7, estimator_7).to(device)

# Defining optimizer
optimizer_7 = torch.optim.Adam(model_7.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername_7 = "./data/deepymod/KDV/lasso_str_tol_1_6"
#shutil.rmtree(foldername)

create_or_reset_directory(foldername_7)

train(model_7,
    train_dataloader_7,
    test_dataloader_7,
    optimizer_7,
    sparsity_scheduler_7,
    log_dir= foldername_7,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=1000,)

"""
tensor([[ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [-1.2937],
        [ 0.0000],
        [-5.5258],
        [ 0.0000],
        [ 4.5286],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [-8.5373]])
"""

########################################################
########################################################


def create_data_8():
    data = loadmat("data/kdv.mat")
    data = scipy.io.loadmat("data/kdv.mat")
    #data = np.load("data/kdv.npy",allow_pickle=True)
    ## preparing and normalizing the input and output data
    t_o = 1 * data["t"].flatten()[0:201, None]
    x_o = 1 * data["x"].flatten()[:, None]
    Exact = np.real(data["usol"])
    deim_instance = DEIM(Exact, 2, t_o.squeeze(), x_o.squeeze(),
                         tolerance = 5 * 1e-7,num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
        
    return coords, data



num_of_samples = 900

dataset_8 = Dataset(
    create_data_8,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)



train_dataloader_8, test_dataloader_8 = get_train_test_loader(
    dataset_8, train_test_split=0.99)


network_8 = NN(2, [32, 32, 32, 32], 1)


library_8 = Library1D(poly_order, diff_order)
estimator_8 = Threshold(0.1)
sparsity_scheduler_8 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
constraint_8 = STRidgeCons()

model_8 = DeepMoD(network_8, library_8, estimator_8, constraint_8, estimator_8).to(device)

# Defining optimizer
optimizer_8 = torch.optim.Adam(model_8.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername_8 = "./data/deepymod/KDV/lasso_str_5_7"
#shutil.rmtree(foldername)

create_or_reset_directory(foldername_8)

train(model_8,
    train_dataloader_8,
    test_dataloader_8,
    optimizer_8,
    sparsity_scheduler_8,
    log_dir= foldername_8,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=1000,)

"""
tensor([[ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [-0.9927],
        [ 0.0000],
        [-5.9780],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000]])
"""

####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################


def create_data_9():
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


num_of_samples = 900

dataset_9 = Dataset(
    create_data_9,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)


train_dataloader_9, test_dataloader_9 = get_train_test_loader(
    dataset_9, train_test_split=0.99)



poly_order = 2
diff_order = 3

n_combinations = (poly_order+1)*(diff_order+1) 
n_features = 1


network_9 = NN(2, [32, 32, 32, 32], 1)

#network = NN(2, [30, 30, 30, 30], 1)

library_9 = Library1D(poly_order, diff_order)
estimator_9 = STRidge()
sparsity_scheduler_9 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
constraint_9 = LeastSquares()

model_9 = DeepMoD(network_9, library_9, estimator_9, constraint_9, estimator_9).to(device)

# Defining optimizer
optimizer_9 = torch.optim.Adam(model_9.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 


foldername_9 = "./data/deepymod/KDV/STR_ols_tol_1_5"
#shutil.rmtree(foldername)

create_or_reset_directory(foldername_9)


train(
    model_9,
    train_dataloader_9,
    test_dataloader_9,
    optimizer_9,
    sparsity_scheduler_9,
    log_dir= foldername_9,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=1000,)

"""
tensor([[ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [-0.9954],
        [ 0.0000],
        [-5.9907],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000]])
"""


###########################################################
###########################################################


def create_data_10():
    data = loadmat("data/kdv.mat")
    data = scipy.io.loadmat("data/kdv.mat")
    #data = np.load("data/kdv.npy",allow_pickle=True)
    ## preparing and normalizing the input and output data
    t_o = 1 * data["t"].flatten()[0:201, None]
    x_o = 1 * data["x"].flatten()[:, None]
    Exact = np.real(data["usol"])
    deim_instance = DEIM(Exact, 2, t_o.squeeze(), x_o.squeeze(),
                         tolerance = 5 * 1e-5,num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
        
    return coords, data



num_of_samples = 900

dataset_10 = Dataset(
    create_data_10,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)



train_dataloader_10, test_dataloader_10 = get_train_test_loader(
    dataset_10, train_test_split=0.99)



network_10 = NN(2, [32, 32, 32, 32], 1)

#network = NN(2, [30, 30, 30, 30], 1)

library_10 = Library1D(poly_order, diff_order)
estimator_10 = STRidge()
sparsity_scheduler_10 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
constraint_10 = LeastSquares()

model_10 = DeepMoD(network_10, library_10, estimator_10, constraint_10, estimator_10).to(device)

# Defining optimizer
optimizer_10 = torch.optim.Adam(model_10.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername_10 = "./data/deepymod/KDV/STR_ols_tol_5_5"
#shutil.rmtree(foldername)

create_or_reset_directory(foldername_10)


train(model_10,
    train_dataloader_10,
    test_dataloader_10,
    optimizer_10,
    sparsity_scheduler_10,
    log_dir= foldername_10,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=1000,)


"""
tensor([[ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [-1.0823],
        [ 0.0000],
        [-5.8526],
        [ 0.0000],
        [ 1.6195],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [-3.2514]])
"""


#######################################################
#######################################################



def create_data_11():
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



num_of_samples = 900

dataset_11 = Dataset(
    create_data_11,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)


train_dataloader_11, test_dataloader_11 = get_train_test_loader(
    dataset_11, train_test_split=0.99)


network_11 = NN(2, [32, 32, 32, 32], 1)

library_11 = Library1D(poly_order, diff_order)
estimator_11 = STRidge() 
sparsity_scheduler_11 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
constraint_11 = LeastSquares()

model_11 = DeepMoD(network_11, library_11, estimator_11, constraint_11, estimator_11).to(device)

# Defining optimizer
optimizer_11 = torch.optim.Adam(model_11.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername_11 = "./data/deepymod/KDV/str_ols_tol_1_6"
#shutil.rmtree(foldername)

create_or_reset_directory(foldername_11)

train(model_11,
    train_dataloader_11,
    test_dataloader_11,
    optimizer_11,
    sparsity_scheduler_11,
    log_dir= foldername_11,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=1000,)

"""
tensor([[ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [-0.9991],
        [ 0.0000],
        [-5.9959],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000]])
"""

########################################################
########################################################


def create_data_12():
    data = loadmat("data/kdv.mat")
    data = scipy.io.loadmat("data/kdv.mat")
    #data = np.load("data/kdv.npy",allow_pickle=True)
    ## preparing and normalizing the input and output data
    t_o = 1 * data["t"].flatten()[0:201, None]
    x_o = 1 * data["x"].flatten()[:, None]
    Exact = np.real(data["usol"])
    deim_instance = DEIM(Exact, 2, t_o.squeeze(), x_o.squeeze(),
                         tolerance = 5 * 1e-7,num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
        
    return coords, data



num_of_samples = 900

dataset_12 = Dataset(
    create_data_12,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)



train_dataloader_12, test_dataloader_12 = get_train_test_loader(
    dataset_12, train_test_split=0.99)


network_12 = NN(2, [32, 32, 32, 32], 1)


library_12 = Library1D(poly_order, diff_order)
estimator_12 = STRidge()
sparsity_scheduler_12 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
constraint_12 = LeastSquares()

model_12 = DeepMoD(network_12, library_12, estimator_12, constraint_12, estimator_12).to(device)

# Defining optimizer
optimizer_12 = torch.optim.Adam(model_12.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername_12 = "./data/deepymod/KDV/str_ols_tol_5_7"
#shutil.rmtree(foldername)

create_or_reset_directory(foldername_12)

train(model_12,
    train_dataloader_12,
    test_dataloader_12,
    optimizer_12,
    sparsity_scheduler_12,
    log_dir= foldername_12,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=1000,)

"""
tensor([[ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [-1.0713],
        [ 0.0000],
        [-5.5245],
        [ 0.0000],
        [ 1.3484],
        [ 0.0000],
        [-2.0393],
        [ 0.0000],
        [-3.5442]])
"""
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################


def create_data_13():
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



num_of_samples = 900

dataset_13 = Dataset(
    create_data_13,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)



train_dataloader_13, test_dataloader_13 = get_train_test_loader(
    dataset_13, train_test_split=0.99)



poly_order = 2
diff_order = 3

n_combinations = (poly_order+1)*(diff_order+1) 
n_features = 1


network_13 = NN(2, [32, 32, 32, 32], 1)

#network = NN(2, [30, 30, 30, 30], 1)

library_13 = Library1D(poly_order, diff_order)
estimator_13 = Threshold(0.1)
sparsity_scheduler_13 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
constraint_13 = LeastSquares()

model_13 = DeepMoD(network_13, library_13, estimator_13, constraint_13, estimator_13).to(device)

# Defining optimizer
optimizer_13 = torch.optim.Adam(model_13.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 


foldername_13 = "./data/deepymod/KDV/lasso_ols_tol_1_5"
#shutil.rmtree(foldername)

create_or_reset_directory(foldername_13)


train(
    model_13,
    train_dataloader_13,
    test_dataloader_13,
    optimizer_13,
    sparsity_scheduler_13,
    log_dir= foldername_13,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=1000,)

"""
tensor([[ 0.0000],
        [ 0.0144],
        [ 0.0000],
        [-1.0074],
        [ 0.0000],
        [-6.0651],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000]])
"""


###########################################################
###########################################################


def create_data_14():
    data = loadmat("data/kdv.mat")
    data = scipy.io.loadmat("data/kdv.mat")
    #data = np.load("data/kdv.npy",allow_pickle=True)
    ## preparing and normalizing the input and output data
    t_o = 1 * data["t"].flatten()[0:201, None]
    x_o = 1 * data["x"].flatten()[:, None]
    Exact = np.real(data["usol"])
    deim_instance = DEIM(Exact, 2, t_o.squeeze(), x_o.squeeze(),
                         tolerance = 5 * 1e-5,num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
        
    return coords, data


num_of_samples = 900

dataset_14 = Dataset(
    create_data_14,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)


train_dataloader_14, test_dataloader_14 = get_train_test_loader(
    dataset_14, train_test_split=0.99)


network_14 = NN(2, [32, 32, 32, 32], 1)

#network = NN(2, [30, 30, 30, 30], 1)

library_14 = Library1D(poly_order, diff_order)
estimator_14 = Threshold(0.1)
sparsity_scheduler_14 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
constraint_14 = LeastSquares()

model_14 = DeepMoD(network_14, library_14, estimator_14, constraint_14, estimator_14).to(device)

# Defining optimizer
optimizer_14 = torch.optim.Adam(model_10.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername_14 = "./data/deepymod/KDV/lasso_ols_tol_5_5"
#shutil.rmtree(foldername)

create_or_reset_directory(foldername_14)


train(model_14,
    train_dataloader_14,
    test_dataloader_14,
    optimizer_14,
    sparsity_scheduler_14,
    log_dir= foldername_14,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=1000,)

"""
tensor([[-1.0551e-03],
        [-2.8053e+00],
        [-1.3475e+00],
        [-2.5160e+00],
        [ 8.9025e-02],
        [-2.2765e+01],
        [ 0.0000e+00],
        [-4.3573e+01],
        [ 5.4488e-01],
        [ 0.0000e+00],
        [ 4.3635e+01],
        [-1.7394e+02]])
"""

#######################################################
#######################################################

def create_data_15():
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



num_of_samples = 900

dataset_15 = Dataset(
    create_data_15,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)


train_dataloader_15, test_dataloader_15 = get_train_test_loader(
    dataset_15, train_test_split=0.99)


network_15 = NN(2, [32, 32, 32, 32], 1)

library_15 = Library1D(poly_order, diff_order)
estimator_15 = Threshold(0.1) 
sparsity_scheduler_15 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
constraint_15 = LeastSquares()

model_15 = DeepMoD(network_15, library_15, estimator_15, constraint_15, estimator_15).to(device)

# Defining optimizer
optimizer_15 = torch.optim.Adam(model_15.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername_15 = "./data/deepymod/KDV/lasso_ols_tol_1_6"
#shutil.rmtree(foldername)

create_or_reset_directory(foldername_15)

train(model_15,
    train_dataloader_15,
    test_dataloader_15,
    optimizer_15,
    sparsity_scheduler_15,
    log_dir= foldername_15,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=1000,)

"""
tensor([[ 0.0000],
        [-0.0095],
        [ 0.0000],
        [-0.9929],
        [ 0.0000],
        [-5.9471],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000]])
"""

########################################################
########################################################

def create_data_16():
    data = loadmat("data/kdv.mat")
    data = scipy.io.loadmat("data/kdv.mat")
    #data = np.load("data/kdv.npy",allow_pickle=True)
    ## preparing and normalizing the input and output data
    t_o = 1 * data["t"].flatten()[0:201, None]
    x_o = 1 * data["x"].flatten()[:, None]
    Exact = np.real(data["usol"])
    deim_instance = DEIM(Exact, 2, t_o.squeeze(), x_o.squeeze(),
                         tolerance = 5 * 1e-7,num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
        
    return coords, data


num_of_samples = 900

dataset_16 = Dataset(
    create_data_16,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)



train_dataloader_16, test_dataloader_16 = get_train_test_loader(
    dataset_16, train_test_split=0.99)

network_16 = NN(2, [32, 32, 32, 32], 1)


library_16 = Library1D(poly_order, diff_order)
estimator_16 = Threshold(0.1)
sparsity_scheduler_16 = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
constraint_16 = LeastSquares()

model_16 = DeepMoD(network_16, library_16, estimator_16, constraint_16, estimator_16).to(device)

# Defining optimizer
optimizer_16 = torch.optim.Adam(model_16.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername_16 = "./data/deepymod/KDV/lasso_ols_tol_5_7"
#shutil.rmtree(foldername)

create_or_reset_directory(foldername_16)

train(model_16,
    train_dataloader_16,
    test_dataloader_16,
    optimizer_16,
    sparsity_scheduler_16,
    log_dir= foldername_16,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=1000,)

"""
tensor([[ 0.0000],
        [-0.0274],
        [ 0.0000],
        [-0.9578],
        [ 0.0000],
        [-5.7962],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [ 0.0000]])
"""

#########################################################
#########################################################


print(model_1.constraint.coeff_vectors[0].detach().cpu())
print(model_2.constraint.coeff_vectors[0].detach().cpu())
print(model_3.constraint.coeff_vectors[0].detach().cpu())
print(model_4.constraint.coeff_vectors[0].detach().cpu())


print(model_5.constraint.coeff_vectors[0].detach().cpu())
print(model_6.constraint.coeff_vectors[0].detach().cpu())
print(model_7.constraint.coeff_vectors[0].detach().cpu())
print(model_8.constraint.coeff_vectors[0].detach().cpu())



print(model_9.constraint.coeff_vectors[0].detach().cpu())
print(model_10.constraint.coeff_vectors[0].detach().cpu())
print(model_11.constraint.coeff_vectors[0].detach().cpu())
print(model_12.constraint.coeff_vectors[0].detach().cpu())


print(model_13.constraint.coeff_vectors[0].detach().cpu())
print(model_14.constraint.coeff_vectors[0].detach().cpu())
print(model_15.constraint.coeff_vectors[0].detach().cpu())
print(model_16.constraint.coeff_vectors[0].detach().cpu())





