#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:56:59 2023

@author: forootani
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:57:23 2023

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


##############################
# Input precision value for the DEIM algorithm

deim_tol = 1 * 1e-7

neu_num = 32

##############################



def create_data():
    data = loadmat("data/kdv.mat")
    
    data = scipy.io.loadmat("data/kdv.mat")
    #data = np.load("data/kdv.npy",allow_pickle=True)

    t_o = 1 * data["t"].flatten()[0:201, None]
    x_o = 1 * data["x"].flatten()[:, None]
    Exact = np.real(data["usol"])
    
    deim_instance = DEIM(Exact, 2, t_o.squeeze(), x_o.squeeze(),
                         tolerance = deim_tol, num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
        
    return coords, data

###############################################################

# Plotting the curve


x_t, u = create_data()
x_t = x_t.detach().cpu().numpy().reshape(-1,2)
u = u.detach().cpu().numpy()
fig, ax = plt.subplots()
im = ax.scatter(x_t[:,0], x_t[:,1], c=u[:,0], marker="x", s=10)
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.colorbar(mappable=im)

plt.show()




num_of_samples = 900


dataset = Dataset(
    create_data,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,)


train_dataloader_1, test_dataloader_1 = get_train_test_loader(
    dataset, train_test_split=0.99)


##########################
##########################

poly_order = 2
diff_order = 3

n_combinations = (poly_order+1)*(diff_order+1) 
n_features = 1


network_1 = NN(2, [neu_num, neu_num, neu_num, neu_num], 1)
library_1 = Library1D(poly_order, diff_order)
sparsity_scheduler_1 = Periodic(periodicity=50, initial_iteration=1000)
constraint_1 = STRidgeCons(tol = 0.1)
estimator_1 = STRidge(lam = 0.0000, maxit = 100, tol = 0.4)
model_1 = DeepMoD(network_1, library_1, estimator_1, constraint_1).to(device)

# Defining optimizer
optimizer_1 = torch.optim.Adam(model_1.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 


foldername_1 = "./data/deepymod/kdv/NN_8"
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
    #patience=1000,
    )


#######################################
#######################################
#######################################

print(torch.round(model_1.constraint.coeff_vectors[0].detach().cpu(), decimals=4))


""" 8
tensor([[  0.0000],
        [  0.2349],
        [  0.0000],
        [ -1.0245],
        [  0.0000],
        [-10.0665],
        [ -0.4588],
        [  3.0480],
        [  0.0000],
        [ 14.0237],
        [  1.0099],
        [ -4.1729]])

16
tensor([[ 0.0000],
        [ 0.0000],
        [ 0.0000],
        [-1.1435],
        [ 0.0000],
        [-5.1284],
        [ 0.0000],
        [ 2.5965],
        [ 0.0000],
        [-3.9851],
        [ 0.0000],
        [-6.9342]])

"""


