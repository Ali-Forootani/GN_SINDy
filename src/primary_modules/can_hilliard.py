#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 08:08:04 2023

@author: forootani
"""

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
from deepymod.model.library import Library2D, Library2D_4O
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

data = np.load("data/CH_Frame1200_X64_Y64_GammaOne_1_GammaTwo_1.npy")



###############################
###############################




def create_data():
    #phi_data = np.load("data/phi_2D_test.npy")
    
    phi_data = np.load("data/CH_Frame1200_X64_Y64_GammaOne_1_GammaTwo_1.npy")
    
    act_num_t = phi_data.shape[0]
    
    phi_data = phi_data [0:10, : , :]
    #phi_data = phi_data [:, :, :]
    
    phi_data = np.transpose(phi_data, (1, 0, 2))
    phi_data = np.transpose(phi_data, (0, 2, 1))
    
    t_num = phi_data.shape[2]
    x_num = phi_data.shape[0]
    y_num = phi_data.shape[1]
    ## preparing and normalizing the input and output data
    t = np.linspace(0, 1, t_num)
    x = np.linspace(-1, 1, x_num)
    y = np.linspace(-1, 1, y_num)
    
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")

    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], T.flatten()[:, None]))

    u_star = phi_data.flatten()[:, None]
    print("The coodinates have shape {}".format(X_star.shape))
    print("The data has shape {}".format(u_star.shape))
    
    
    X_star = torch.from_numpy(X_star)
    u_star = torch.from_numpy(u_star)
    
   
    return X_star, u_star



coords, data = create_data()





fig, axes = plt.subplots(ncols=1, figsize=(6, 4))

im0 = axes.scatter(coords[:,1], coords[:,0], c= data[:,0], cmap='coolwarm', marker="+")
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('t = 0')



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(coords[:, 0], coords[:, 1], data[:, 0], cmap='nearest', marker='o')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('t = 0')

plt.show()





dataset = Dataset(
    create_data,
    shuffle = False,
    preprocess_kwargs={
        "noise_level": 0.000,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": 50000},
    device=device,
)


new_coords = dataset.coords.detach().cpu().numpy()
new_data = dataset.data.detach().cpu().numpy()


fig, axes = plt.subplots(ncols=1, figsize=(6, 4))

im0 = axes.scatter(new_coords[:2000,1], new_coords[:2000,0], c= new_data[:2000], cmap='coolwarm', marker="+")
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('t = 0')


###########################
###########################

train_dataloader, test_dataloader = get_train_test_loader(dataset, train_test_split=0.8)

network = NN(3, [128, 128, 128, 128], 1)
library = Library2D_4O(poly_order=2)

estimator = Threshold(0.01)
estimator_2 = STRidge(lam = 0.00001, maxit = 100, tol = 0.1)

sparsity_scheduler = TrainTestPeriodic()

constraint = LeastSquares()
constraint_2 = STRidgeCons()
constraint_3 = Ridge()

model = DeepMoD(network, library, estimator, constraint, estimator).to(device)

# Defining optimizer
optimizer = torch.optim.Adam(
    model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-4
)

train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    sparsity_scheduler,
    log_dir="runs/2DAD/",
    max_iterations=20000,
    delta = 1e-5,
    patience= 1000,
    write_iterations=25,
)

print(torch.round(model.constraint.coeff_vectors[0].detach().cpu()))


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
