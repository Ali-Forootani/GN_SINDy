#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:16:40 2023

@author: forootani
"""


import sys
import os
import scipy.io as sio

cwd = os.getcwd()
sys.path.append(cwd)


import numpy as np
import torch
import matplotlib.pylab as plt

# DeepMoD functions

from deepymod import DeepMoD
from deepymod.data import Dataset, get_train_test_loader
from deepymod.data.samples import Subsample_random
from deepymod.model.func_approx import NN
from deepymod.model.library import Library2D
from deepymod.model.constraint import LeastSquares, Ridge, STRidgeCons
from deepymod.model.sparse_estimators import Threshold, PDEFIND, STRidge
from deepymod.training import train
from deepymod.training.sparsity_scheduler import TrainTestPeriodic
from scipy.io import loadmat

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)

# Configuring GPU or CPU

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)


################################
################################

data = loadmat("data/advection_diffusion.mat")
usol = np.real(data["Expression1"]).astype("float32")
usol = usol.reshape((51, 51, 61, 4))
x_v= usol[:,:,:,0]
y_v = usol[:,:,:,1]
t_v = usol[:,:,:,2]
u_v = usol[:,:,:,3]

fig, axes = plt.subplots(ncols=3, figsize=(15, 4))

im0 = axes[0].contourf(x_v[:,:,60], y_v[:,:,60], u_v[:,:,60], cmap='coolwarm')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('t = 60')

im1 = axes[1].contourf(x_v[:,:,0], y_v[:,:,0], u_v[:,:,0], cmap='coolwarm')
axes[1].set_xlabel('x')
axes[1].set_title('t = 0')

im2 = axes[2].contourf(x_v[:,:,10], y_v[:,:,10], u_v[:,:,10], cmap='coolwarm')
axes[2].set_xlabel('x')
axes[2].set_title('t= 10')

fig.colorbar(im1, ax=axes.ravel().tolist())
plt.show()

###############################
###############################


def create_data():
    data = loadmat("data/advection_diffusion.mat")
    usol = np.real(data["Expression1"]).astype("float32")
    
    print(usol.shape)
    
    usol = usol.reshape((51, 51, 61, 4))
    x_v = usol[:,:,:,0]
    y_v = usol[:,:,:,1]
    t_v = usol[:,:,:,2]
    u_v = usol[:,:,:,3]
    coords = torch.from_numpy(np.stack((t_v,x_v, y_v), axis=-1))
    data = torch.from_numpy(usol[:, :, :, 3]).unsqueeze(-1)
    # alternative way of providing the coordinates
    # coords = torch.from_numpy(np.transpose((t_v.flatten(), x_v.flatten(), y_v.flatten())))
    # data = torch.from_numpy(usol[:, :, :, 3].reshape(-1,1))
    print("The coodinates have shape {}".format(coords.shape))
    print("The data has shape {}".format(data.shape))
    return coords, data


coords, data = create_data()


dataset = Dataset(
    create_data,
    preprocess_kwargs={
        "noise_level": 0.000,
        "normalize_coords": True,
        "normalize_data": True,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": 100},
    device=device,
)



###########################
###########################



train_dataloader, test_dataloader = get_train_test_loader(dataset, train_test_split=0.8)

network = NN(3, [50, 50, 50, 50], 1)
library = Library2D(poly_order=1)

estimator = Threshold(0.1)
estimator_2 = STRidge(lam = 0.00001, maxit = 100, tol = 0.1)

sparsity_scheduler = TrainTestPeriodic()

constraint = LeastSquares()
constraint_2 = STRidgeCons()
constraint_3 = Ridge()

model = DeepMoD(network, library, estimator_2, constraint_2, estimator).to(device)

# Defining optimizer
optimizer = torch.optim.Adam(
    model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3
)

train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    sparsity_scheduler,
    log_dir="runs/2DAD/",
    max_iterations=25000,
    delta = 1e-5,
    patience=200,
)

print(model.constraint.coeff_vectors[0].detach().cpu())


"""
[tensor([[-0.3593],
         [ 0.1382],
         [ 0.2658],
         [ 0.0000],
         [ 0.0000],
         [ 0.0000],
         [-0.4539],
         [ 0.0000],
         [ 0.0000],
         [ 0.0000],
         [ 0.0000],
         [ 0.0000]], device='cuda:0', grad_fn=<MaskedScatterBackward0>)]
"""
