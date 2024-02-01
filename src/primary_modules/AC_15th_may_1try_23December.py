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



def create_data():
    data = loadmat("data/AC.mat")
    
    data = scipy.io.loadmat("data/AC.mat")

    ## preparing and normalizing the input and output data
    t_o = 10*data["tt"].flatten()[0:201, None]
    x_o = 5*data["x"].flatten()[:, None]
    Exact = np.real(data["uu"])
    #X, T = np.meshgrid(x, t, indexing="ij")
    #X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    #u_star = Exact.flatten()[:, None]
    
    print(t_o.squeeze().shape)
    
    deim_instance = DEIM(Exact, 3, t_o.squeeze(), x_o.squeeze(), num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
    #coords = torch.from_numpy(np.stack((t,x), axis=-1))
    #data = torch.from_numpy(np.real(data["uu"])).unsqueeze(-1)
    # alternative way of providing the coordinates
    # coords = torch.from_numpy(np.transpose((t_v.flatten(), x_v.flatten(), y_v.flatten())))
    # data = torch.from_numpy(usol[:, :, :, 3].reshape(-1,1))
    #print("The coodinates have shape {}".format(X_star.shape))
    #print("The data has shape {}".format(u_star.shape))
    #X_star = torch.tensor(X_star, dtype= float, )
    #u_star = torch.tensor(u_star, dtype= float,)
    
    return coords, data


x_t, u = create_data()

x_t = x_t.detach().cpu().numpy().reshape(-1,2)
u = u.detach().cpu().numpy()
fig, ax = plt.subplots()
im = ax.scatter(x_t[:,1], x_t[:,0], c=u[:,0], marker="x", s=10)
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.colorbar(mappable=im)




dataset = Dataset(
    create_data,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": True,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": 100},
    device=device,)




coords = dataset.get_coords().detach().cpu()
data = dataset.get_data().detach().cpu()

fig, ax = plt.subplots()
im = ax.scatter(coords[:,0], coords[:,1], c=data[:,0], marker="x", s=10)
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.colorbar(mappable=im)

plt.show()




train_dataloader, test_dataloader = get_train_test_loader(
    dataset, train_test_split=0.99)


##########################
##########################

poly_order = 3
diff_order = 2

n_combinations = (poly_order+1)*(diff_order+1) 
n_features = 1


network = NN(2, [64, 64, 64, 64], 1)

library = Library1D(poly_order, diff_order)
estimator = Threshold(0.1) 
sparsity_scheduler = TrainTestPeriodic(periodicity=200, patience=1000, delta=1e-5)
constraint = LeastSquares()
constraint_2 = Ridge()
constraint_3 = STRidgeCons()

estimator_2 = STRidge()

#linear_module = CoeffsNetwork(int(n_combinations),int(n_features))


#constraint = Ridge()
# Configuration of the sparsity scheduler
model = DeepMoD(network, library, estimator, constraint, estimator).to(device)


# Defining optimizer
optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 





train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    sparsity_scheduler,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=200,)



model.sparsity_masks

print(model.sparsity_masks)

print(model.estimator_coeffs())

print(model.constraint.coeff_vectors)











