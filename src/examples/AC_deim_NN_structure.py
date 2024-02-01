#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:52:45 2024

@author: forootani

Allen-Cahn GN-SINDy discovery analysis with different neural network structure.

aliforootani@ieee.org
forootani@mpi-magdeburg.mpg.de

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



def create_data_1():
    data = loadmat( root_dir + "/src/data/AC.mat")
    data = scipy.io.loadmat( root_dir + "/src/data/AC.mat")

    ## preparing and normalizing the input and output data
    t_o = 1*data["tt"].flatten()[0:201, None]
    x_o = 1*data["x"].flatten()[:, None]
    Exact = np.real(data["uu"])
    deim_instance = DEIM(Exact, 3, t_o.squeeze(), x_o.squeeze(), 
                         tolerance = 1e-7, num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
    return coords, data


x_t, u = create_data_1()

###################################################
###################################################

num_of_samples = 120


dataset_1 = Dataset(
    create_data_1,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,},
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)


coords_1 = dataset_1.get_coords().detach().cpu()
data_1 = dataset_1.get_data().detach().cpu()

train_dataloader_1, test_dataloader_1 = get_train_test_loader(dataset_1,
                                                              train_test_split=0.99)

poly_order = 3
diff_order = 3
n_combinations = (poly_order+1)*(diff_order+1) 
n_features = 1
network_1 = NN(2, [64, 64, 64, 64], 1)
library_1 = Library1D(poly_order, diff_order)
sparsity_scheduler_1 = TrainTestPeriodic(periodicity=100, patience=1000, delta=1e-5)
#sparsity_scheduler_1 = Periodic()
constraint_1 = STRidgeCons()
estimator_1 = STRidge(tol=0.8)

model_1 = DeepMoD(network_1, library_1, estimator_1, constraint_1).to(device)

# Defining optimizer
optimizer_1 = torch.optim.Adam(model_1.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername_1 = "./data/deepymod/Allen_Cahn/NN_64"
create_or_reset_directory(foldername_1)



train(model_1, train_dataloader_1,
    test_dataloader_1,
    optimizer_1,
    sparsity_scheduler_1,
    log_dir= foldername_1,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=35000,
    delta=1e-4,
    patience=200,)



##########################################################
##########################################################






##########################################################
##########################################################


def create_data_2():
    data = loadmat(root_dir + "/src/data/AC.mat")
    data = scipy.io.loadmat(root_dir + "/src/data/AC.mat")

    ## preparing and normalizing the input and output data
    t_o = 1*data["tt"].flatten()[0:201, None]
    x_o = 1*data["x"].flatten()[:, None]
    Exact = np.real(data["uu"])
    deim_instance = DEIM(Exact, 3, t_o.squeeze(), x_o.squeeze(), 
                         tolerance = 1e-7, num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
    return coords, data

x_t, u = create_data_2()


dataset_2 = Dataset(
    create_data_2,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,},
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)



train_dataloader_2, test_dataloader_2 = get_train_test_loader(dataset_1,
                                                              train_test_split=0.99)




network_2 = NN(2, [32, 32, 32, 32], 1)
library_2 = Library1D(poly_order, diff_order)
sparsity_scheduler_2 = TrainTestPeriodic(periodicity=100, patience=1000, delta=1e-5)
#sparsity_scheduler_2 = Periodic()


constraint_2 = STRidgeCons()
estimator_2 = STRidge(tol=0.8)

model_2 = DeepMoD(network_2, library_2, estimator_2, constraint_2).to(device)

# Defining optimizer
optimizer_2 = torch.optim.Adam(model_2.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername_2 = "./data/deepymod/Allen_Cahn/NN_32"
create_or_reset_directory(foldername_2)



train(model_2, train_dataloader_1,
    test_dataloader_2,
    optimizer_2,
    sparsity_scheduler_2,
    log_dir= foldername_2,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=35000,
    delta=1e-4,
    patience=200,)



##############################################################################
##############################################################################

"""

def create_data_3():
    data = loadmat(root_dir +"/src/data/AC.mat")
    data = scipy.io.loadmat(root_dir +"/src/data/AC.mat")

    ## preparing and normalizing the input and output data
    t_o = 1*data["tt"].flatten()[0:201, None]
    x_o = 1*data["x"].flatten()[:, None]
    Exact = np.real(data["uu"])
    deim_instance = DEIM(Exact, 3, t_o.squeeze(), x_o.squeeze(), 
                         tolerance = 1e-7, num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
    return coords, data

x_t, u = create_data_3()



dataset_3 = Dataset(
    create_data_3,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,},
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)



train_dataloader_3, test_dataloader_3 = get_train_test_loader(dataset_3,
                                                              train_test_split=0.99)



network_3 = NN(2, [16, 16, 16, 16], 1)
library_3 = Library1D(poly_order, diff_order)
#sparsity_scheduler_3 = TrainTestPeriodic(periodicity=100, patience=1000, delta=1e-5)
sparsity_scheduler_3 = Periodic()
constraint_3 = STRidgeCons()
estimator_3 = STRidge(tol=0.1)

model_3 = DeepMoD(network_3, library_3, estimator_3, constraint_3).to(device)

# Defining optimizer
optimizer_3 = torch.optim.Adam(model_3.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername_3 = "./data/deepymod/Allen_Cahn/NN_16"
create_or_reset_directory(foldername_3)



train(model_3, train_dataloader_1,
    test_dataloader_3,
    optimizer_3,
    sparsity_scheduler_3,
    log_dir= foldername_3,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=35000,
    delta=1e-4,
    patience=200,)

"""

#######################################################
#######################################################


"""
def create_data_4():
    data = loadmat(root_dir +"/src/data/AC.mat")
    data = scipy.io.loadmat(root_dir + "/src/data/AC.mat")

    ## preparing and normalizing the input and output data
    t_o = 1*data["tt"].flatten()[0:201, None]
    x_o = 1*data["x"].flatten()[:, None]
    Exact = np.real(data["uu"])
    deim_instance = DEIM(Exact, 3, t_o.squeeze(), x_o.squeeze(), 
                         tolerance = 1e-7, num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
    return coords, data

x_t, u = create_data_4()

x_t = x_t.detach().cpu().numpy().reshape(-1,2)
u = u.detach().cpu().numpy()
fig, ax = plt.subplots()
im = ax.scatter(x_t[:,0], x_t[:,1], c=u[:,0], marker="x", s=10)
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.colorbar(mappable=im)




dataset_4 = Dataset(
    create_data_4,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,},
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)




train_dataloader_4, test_dataloader_4 = get_train_test_loader(dataset_4,
                                                              train_test_split=0.99)

network_4 = NN(2, [8, 8, 8, 8], 1)
library_4 = Library1D(poly_order, diff_order)
#sparsity_scheduler_4 = TrainTestPeriodic(periodicity=100, patience=1000, delta=1e-5)
sparsity_scheduler_4 = Periodic()


constraint_4 = STRidgeCons()
estimator_4 = STRidge(tol=0.1)

model_4 = DeepMoD(network_4, library_4, estimator_4, constraint_4).to(device)

# Defining optimizer
optimizer_4 = torch.optim.Adam(model_4.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername_4 = "./data/deepymod/Allen_Cahn/tolerance_8"
create_or_reset_directory(foldername_4)



train(model_4, train_dataloader_4,
    test_dataloader_4,
    optimizer_4,
    sparsity_scheduler_4,
    log_dir= foldername_4,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=35000,
    delta=1e-4,
    patience=200,)

"""

#######################################################
#######################################################

print(torch.round(model_1.constraint.coeff_vectors[0].detach().cpu(), decimals=4))
print(torch.round(model_2.constraint.coeff_vectors[0].detach().cpu(), decimals=4))
#print(torch.round(model_3.constraint.coeff_vectors[0].detach().cpu(), decimals=4))
#print(torch.round(model_4.constraint.coeff_vectors[0].detach().cpu(), decimals=4))




