#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 14:16:15 2023

@author: forootani
"""

import matplotlib.pyplot as plt

# General imports
import numpy as np
import torch

# DeePyMoD imports
from deepymod import DeepMoD
from deepymod.data import Dataset, get_train_test_loader
from deepymod.data.samples import Subsample_random
from deepymod.data.burgers import burgers_delta
from deepymod.model.constraint import LeastSquares
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D
from deepymod.model.sparse_estimators import Threshold
from deepymod.training import train
from deepymod.training.sparsity_scheduler import Periodic, TrainTest, TrainTestPeriodic
from scipy.io import loadmat
from deepymod.data.DEIM_class import DEIM




if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




def create_data():
    
    # Load and plot the data
    data = loadmat("data/kuramoto_sivishinky.mat")
    t_o = np.ravel(data["tt"])
    x_o = np.ravel(data["x"])
    Exact = data["uu"]
    dt = time[1] - time[0]
    dx = x[1] - x[0]
    

    ## preparing and normalizing the input and output data
    #t_o = data["t"].flatten()[0:201, None]
    #x_o = data["x"].flatten()[:, None]
    #Exact = np.real(data["usol"])
    #X, T = np.meshgrid(x, t, indexing="ij")
    #X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    #u_star = Exact.flatten()[:, None]
    
    
    
    deim_instance = DEIM(Exact, 4, t_o.squeeze(), x_o.squeeze(),
                         tolerance = 1e-4, num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
        
    
    
    
    return coords, data






# Load and plot the data
data = loadmat("data/kuramoto_sivishinky.mat")
time = np.ravel(data["tt"])
x = np.ravel(data["x"])
u = data["uu"]
dt = time[1] - time[0]
dx = x[1] - x[0]




x_t, u = create_data()
x_t = x_t.detach().cpu().numpy().reshape(-1,2)
u = u.detach().cpu().numpy()
fig, ax = plt.subplots()
im = ax.scatter(x_t[:,0], x_t[:,1], c=u[:,0], marker="x", s=10)
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.colorbar(mappable=im)


dataset = Dataset(
    create_data,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": 4000},
    device=device,)




coords = dataset.get_coords().detach().cpu()
data = dataset.get_data().detach().cpu()

fig, ax = plt.subplots()
im = ax.scatter(coords[:,0], coords[:,1], c=data[:,0], marker="x", s=10)
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.colorbar(mappable=im)

plt.show()


train_dataloader, test_dataloader = get_train_test_loader(dataset, train_test_split=0.99)

network = NN(2, [64, 64, 64, 64], 1)


library = Library1D(poly_order=2, diff_order=4)


estimator = Threshold(0.1)
sparsity_scheduler = TrainTestPeriodic(periodicity=500, patience=1000, delta=1e-5)


constraint = LeastSquares()

model = DeepMoD(network, library, estimator, constraint, estimator).to(device)

# Defining optimizer
optimizer = torch.optim.Adam(
    model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3)



foldername = "./data/deepymod/kuramoto_sivashinsky/"
train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    sparsity_scheduler,
    log_dir=foldername,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=10000,
)



model.sparsity_masks

print(model.sparsity_masks)

print(model.estimator_coeffs())

print(model.constraint.coeff_vectors)



#######################################


from deepymod.analysis import load_tensorboard

history = load_tensorboard(foldername)
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

for history_key in history.keys():
    history_key_parts = history_key.split("_")
    if history_key_parts[0] == "loss":
        if history_key_parts[-1] == "0":
            axs[0].semilogy(
                history[history_key],
                label=history_key_parts[1] + "_" + history_key_parts[-1],
                linestyle="--",
            )
        elif history_key_parts[-1] == "1":
            axs[0].semilogy(
                history[history_key],
                label=history_key_parts[1] + "_" + history_key_parts[-1],
                linestyle=":",
            )
        else:
            axs[0].semilogy(
                history[history_key],
                label=history_key_parts[1] + "_" + history_key_parts[-1],
                linestyle="-",
            )
        if history_key_parts[0] == "remaining":
            axs[0].semilogy(
                history[history_key],
                label=history_key_parts[1]
                + "_"
                + history_key_parts[3]
                + "_"
                + history_key_parts[4],
                linestyle="-.",
            )
    if history_key_parts[0] == "coeffs":
        if history_key_parts[2] == "0":
            axs[1].plot(
                history[history_key],
                label=history_key_parts[2]
                + "_"
                + history_key_parts[3]
                + "_"
                + history_key_parts[4],
                linestyle="--",
            )
        elif history_key_parts[2] == "1":
            axs[1].plot(
                history[history_key],
                label=history_key_parts[2]
                + "_"
                + history_key_parts[3]
                + "_"
                + history_key_parts[4],
                linestyle=":",
            )
        else:
            axs[1].plot(
                history[history_key],
                label=history_key_parts[2]
                + "_"
                + history_key_parts[3]
                + "_"
                + history_key_parts[4],
                linestyle="-",
            )
    if history_key_parts[0] == "unscaled":
        if history_key_parts[3] == "0":
            axs[2].plot(
                history[history_key],
                label=history_key_parts[3]
                + "_"
                + history_key_parts[4]
                + "_"
                + history_key_parts[5],
                linestyle="--",
            )
        elif history_key_parts[3] == "1":
            axs[2].plot(
                history[history_key],
                label=history_key_parts[3]
                + "_"
                + history_key_parts[4]
                + "_"
                + history_key_parts[5],
                linestyle=":",
            )
        else:
            axs[2].plot(
                history[history_key],
                label=history_key_parts[3]
                + "_"
                + history_key_parts[4]
                + "_"
                + history_key_parts[5],
                linestyle="-",
            )
    if history_key_parts[0] == "estimator":
        if history_key_parts[3] == "0":
            axs[3].plot(
                history[history_key],
                label=history_key_parts[3]
                + "_"
                + history_key_parts[4]
                + "_"
                + history_key_parts[5],
                linestyle="--",
            )
        elif history_key_parts[3] == "1":
            axs[3].plot(
                history[history_key],
                label=history_key_parts[3]
                + "_"
                + history_key_parts[4]
                + "_"
                + history_key_parts[5],
                linestyle=":",
            )
        else:
            axs[3].plot(
                history[history_key],
                label=history_key_parts[3]
                + "_"
                + history_key_parts[4]
                + "_"
                + history_key_parts[5],
                linestyle="-",
            )

# axs[0].set_ylim([-2, 2])
axs[1].set_ylim([-2, 2])
axs[2].set_ylim([-2, 2])
axs[3].set_ylim([-2, 2])

axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[3].legend()

plt.show()






















