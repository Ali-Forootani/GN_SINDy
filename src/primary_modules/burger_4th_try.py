#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 08:48:12 2023

@author: forootani
"""


import numpy as np
import torch

import sys
import os
import scipy.io as sio


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
from deepymod.data.burgers import burgers_delta, burgers_delta_org
from deepymod.model.constraint import LeastSquares, Ridge, STRidgeCons
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D
from deepymod.model.sparse_estimators import Threshold, STRidge
from deepymod.training import train
#from deepymod.training.training_2 import train

from deepymod.training.sparsity_scheduler import Periodic, TrainTest, TrainTestPeriodic

#from deepymod.data.data_set_preparation import DatasetPDE, pde_data_loader



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(10)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#########################
#########################
#########################



# loading the data
#data = sio.loadmat(os.path.dirname( os.path.abspath('') ) +'/Datasets/burgers.mat')



#########################
#########################
#########################

# Making dataset
v = 0.1
A = 1.0

x = torch.linspace(-8, 8, 100)
t = torch.linspace(0.5, 10.0, 100)


#x = torch.tensor(x)
#t = torch.tensor(t)


load_kwargs = {"x": x, "t": t, "v": v, "A": A}
preprocess_kwargs = {"noise_level": 0.00}

#########################
#########################
#########################



num_of_samples = 50


#########################
#########################
#########################

dataset = Dataset(
    burgers_delta,
    load_kwargs=load_kwargs,
    preprocess_kwargs=preprocess_kwargs,
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,
)

coords = dataset.get_coords().cpu()
data = dataset.get_data().cpu()
fig, ax = plt.subplots()
im = ax.scatter(coords[:,0], coords[:,1], c=data[:,0], marker="x", s=10)
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.colorbar(mappable=im)

plt.show()


##########################
##########################


train_dataloader, test_dataloader = get_train_test_loader(
    dataset, train_test_split=1)



##########################
##########################

poly_order = 2
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


##############################################
##############################################
##############################################

import shutil

foldername = "./data/deepymod/burgers_random/"
#shutil.rmtree(foldername)

#######################################
#######################################
#######################################

def create_or_reset_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # If it exists, remove it
        try:
            #os.rmdir(directory_path)
            shutil.rmtree(directory_path)
            print(f"Directory '{directory_path}' already exist so it is removed.")
            #os.rmdir(directory_path)
            #os.rmdir(directory_path)
            print(f"Directory '{directory_path}' is created.")
        except OSError as e:
            print(f"Error removing directory '{directory_path}': {e}")
            return

    # Create the directory
    try:
        os.mkdir(directory_path)
        print(f"Directory '{directory_path}' created.")
    except OSError as e:
        print(f"Error creating directory '{directory_path}': {e}")

# Example usage:
#directory_path = "/path/to/your/directory"
create_or_reset_directory(foldername)







##############################################
##############################################
##############################################

train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    sparsity_scheduler,
    log_dir= foldername,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=200,
)

model.sparsity_masks

print(model.estimator_coeffs())
print(model.constraint.coeff_vectors[0].detach().cpu())


"""
data_2 = sio.loadmat(cwd + '/Datasets/burgers.mat')

u = np.real(data_2['usol'])
x = np.real(data_2['x'][0])
t = np.real(data_2['t'][:,0])
dt = t[1]-t[0]
dx = x[2]-x[1]

X, T = np.meshgrid(x, t)

x = torch.reshape(torch.tensor(X.flatten()),(-1,1))
t = torch.reshape(torch.tensor(T.flatten()),(-1,1))

u_nn = torch.reshape(torch.tensor(u.flatten()),(-1,1))

data_input = torch.cat((t, x),1)


data_input, u_cuda = data_input.to(device), u_nn.to(device)



train_dataloader = pde_data_loader(data_input, u_cuda, batch_size = 100000,
                                   split=0.4, shuffle=True)
"""


"""
from deepymod.analysis import load_tensorboard
from deepymod.utils import plot_config_file
from deepymod.analysis import load_tensorboard
history = load_tensorboard(foldername)
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
line_width=2

for history_key in history.keys():
    history_key_parts = history_key.split("_")
    
    if history_key_parts[0] == "loss":
        if history_key_parts[-1] == "0":
            axs[0].loglog(
                history[history_key],
                label=history_key_parts[1] + "_" + history_key_parts[-1],
                linestyle="--",
                linewidth= line_width
            )
        elif history_key_parts[-1] == "1":
            axs[0].loglog(
                history[history_key],
                label=history_key_parts[1] + "_" + history_key_parts[-1],
                linestyle=":",
                linewidth= line_width
            )
        else:
            axs[0].loglog(
                history[history_key],
                label=history_key_parts[1] + "_" + history_key_parts[-1],
                linestyle="-",
                linewidth= line_width
            )
        if history_key_parts[0] == "remaining":
            axs[0].loglog(
                history[history_key],
                label=history_key_parts[1]
                + "_"
                + history_key_parts[3]
                + "_"
                + history_key_parts[4],
                linestyle="-.",
                linewidth= line_width
            )
    
      
    if history_key_parts[0] == "estimator":
        if history_key_parts[3] == "0":
            
           
            
            if history_key_parts[5] != "2" and history_key_parts[5] != "4":
                
                axs[1].semilogx(
                history[history_key].loc[100:10000],
                label=
                history_key_parts[4]
                + "_"
                + history_key_parts[5],
                linestyle="--",
                linewidth= line_width
                )
        
            elif history_key_parts[5] == "2":
            
            
                axs[1].semilogx(
                    history[history_key].loc[100:10000],
                    label=
                    history_key_parts[4]
                    + "_"
                    + history_key_parts[5],
                    linestyle="-",
                    linewidth= line_width + 2
                    )
            elif history_key_parts[5] == "4":
            
            
                axs[1].semilogx(
                    history[history_key].loc[100:10000],
                    label=
                    history_key_parts[4]
                    + "_"
                    + history_key_parts[5],
                    linestyle="-",
                    linewidth= line_width + 2
                    ) 
            
        
        elif history_key_parts[3] == "1":
            axs[1].plot(
                history[history_key],
                label=
                history_key_parts[4]
                + "_"
                + history_key_parts[5],
                linestyle=":",
                linewidth= line_width
            )
        else:
            axs[1].plot(
                history[history_key],
                label=
                history_key_parts[4]
                + "_"
                + history_key_parts[5],
                linestyle="-",
                linewidth= line_width
            )
            
    
axs[0].set_ylim([0, 100])
axs[1].set_ylim([-2, 2])
#axs[2].set_ylim([-2, 2])
#axs[3].set_ylim([-2, 2])

#axs[0].legend()
#axs[1].legend()
#axs[2].legend()
#axs[3].legend()

# Add legends to the right of the subplots
axs[0].legend(loc='center left', bbox_to_anchor=(0.45, 0.9), ncol=2)
axs[1].legend(loc='center left', bbox_to_anchor=(0.45, 0.85), ncol=3)

axs[0].grid(True)
axs[1].grid(True)

plt.show()

"""



from deepymod.utils import plot_config_file
from deepymod.analysis import load_tensorboard
history = load_tensorboard(foldername)
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
line_width=2

for history_key in history.keys():
    history_key_parts = history_key.split("_")
      
    if history_key_parts[0] == "estimator":
        if history_key_parts[3] == "0":
            
            if history_key_parts[5] != "2" and history_key_parts[5] != "4":
                
                axs[0].semilogx(
                history[history_key].loc[100:],
                
                linestyle="--",
                linewidth= line_width
                )
        
            elif history_key_parts[5] == "2":
                    
                axs[0].semilogx(
                    history[history_key].loc[100:],
                    label= "\r $u_{xx}$",
                    
                    linewidth= line_width + 1
                    )
            elif history_key_parts[5] == "4":
            
                axs[0].semilogx(
                    history[history_key].loc[100:],
                    label= "\r $uu_x$",
                    
                    linewidth= line_width + 1
                    ) 
            
            
    
axs[0].set_ylim([-2.5, 2.5])
axs[0].set_xlabel("Iterations")
axs[0].set_ylabel("Coefficients")


# Add legends to the right of the subplots
#axs[0].legend(loc='center left', bbox_to_anchor=(0.45, 0.9), ncol=2)
#axs[0].legend(loc='center left', bbox_to_anchor=(0.0, 1.05), ncol=2)
#axs[0].grid(True)
axs[0].grid(True)


coords = dataset.get_coords().cpu()
data = dataset.get_data().cpu()

im = axs[1].scatter(coords[:,0], coords[:,1], c=data[:,0], marker="x", 
                    label=r"Random samples",
                    s=20)
axs[1].set_xlabel(r'$t$')
axs[1].set_ylabel(r'$x$', labelpad=0)
axs[1].set_ylim([-8,8])
#axs[1].legend(loc='center left', bbox_to_anchor=(0.0, 1.05), ncol=1)
#fig.colorbar(mappable=im)


##################################
##################################
##################################

x_o = torch.linspace(-8, 8, 100)
t_o = torch.linspace(0.5, 10.0, 100)
v = 0.1
A = 1.0
coords_org , Exact = burgers_delta_org( x_o, t_o, v, A)
im = axs[2].scatter(coords_org[:,0], coords_org[:,1], c=Exact[:,:], marker="x", s=10)
axs[2].set_xlabel(r'$t$')
axs[2].set_ylabel(r'$x$', labelpad=0)
axs[2].set_ylim([-8,8])
fig.colorbar(mappable=im)


#################################
#################################
#################################


fig.legend(
    
    loc="center",  # Change the location to upper center
    ncol=4,
    bbox_to_anchor=(0.51, 1),  # Adjust the coordinates
    bbox_transform=fig.transFigure,
    fontsize=20,
    frameon=True,
)


plt.savefig(foldername+
     "Burgers" + "coefficients" + "random_sampling" + f'{num_of_samples}' +".png", bbox_inches='tight',
    dpi=600,
)
plt.savefig(foldername+
     "Burgers"+ "coefficients" + "random_sampling" + f'{num_of_samples}' + ".pdf", bbox_inches='tight',
    dpi=600,
)

plt.show()




















