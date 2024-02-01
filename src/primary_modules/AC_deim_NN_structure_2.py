#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 08:57:24 2024

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
from deepymod.data.burgers import burgers_delta
from deepymod.model.constraint import LeastSquares, Ridge, STRidgeCons
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D
from deepymod.model.sparse_estimators import Threshold, STRidge
from deepymod.training import train
#from deepymod.training.training_2 import train
from deepymod.training.sparsity_scheduler import Periodic, TrainTest, TrainTestPeriodic
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
from deepymod.data.DEIM_class import DEIM
from deepymod.utils.utilities import create_or_reset_directory

import shutil


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
    data = loadmat("data/AC.mat")
    data = scipy.io.loadmat("data/AC.mat")

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


neu_num=8

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
network_1 = NN(2, [neu_num, neu_num, neu_num, neu_num], 1)
library_1 = Library1D(poly_order, diff_order)
sparsity_scheduler_1 = Periodic()

constraint_1 = STRidgeCons()
estimator_1 = STRidge(tol=0.1)

model_1 = DeepMoD(network_1, library_1, estimator_1, constraint_1).to(device)

# Defining optimizer
optimizer_1 = torch.optim.Adam(model_1.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 

foldername_1 = "./data/deepymod/Allen_Cahn/NN8_tolerance_7"
create_or_reset_directory(foldername_1)



train(model_1, train_dataloader_1,
    test_dataloader_1,
    optimizer_1,
    sparsity_scheduler_1,
    log_dir= foldername_1,
    exp_ID="Test",
    write_iterations=25,
    max_iterations = 35000,
    delta=1e-4,
    patience=200,)



print(torch.round(model_1.constraint.coeff_vectors[0].detach().cpu(), decimals=4))


