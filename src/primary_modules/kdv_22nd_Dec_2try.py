#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 12:29:02 2023

@author: forootani
"""

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

cwd = os.getcwd()
sys.path.append(cwd)

def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir
root_dir = setting_directory(1)

import matplotlib.pyplot as plt

# General imports
import numpy as np
import torch

# DeePyMoD imports
from deepymod import DeepMoD
from deepymod.data import Dataset, get_train_test_loader
from deepymod.data.samples import Subsample_random
from deepymod.data.burgers import burgers_delta
from deepymod.data.kdv import single_soliton_org, double_soliton_org
from deepymod.model.constraint import LeastSquares, Ridge, STRidgeCons
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D
from deepymod.model.sparse_estimators import Threshold, STRidge
from deepymod.training import train
from deepymod.training.sparsity_scheduler import Periodic, TrainTest, TrainTestPeriodic
from deepymod.utils.utilities import create_or_reset_directory

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
    #data = loadmat("data/kdv.mat")
    
    
    
    #data = scipy.io.loadmat("data/kdv.mat")
    data = np.load("data/kdv.npy",allow_pickle=True).item()

    ## preparing and normalizing the input and output data
    t_o = data["t"].flatten()[0:201, None]
    x_o = data["x"][:,0].flatten()[:, None]
    Exact = np.real(data["u"])
    X, T = np.meshgrid(x_o, t_o, indexing="ij")
    coords = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    data = Exact.flatten()[:, None]
    
    
    
    coords = torch.tensor(coords, dtype= float, )
    data = torch.tensor(data, dtype= float,)
    
    return coords, data


x_t, u = create_data()
x_t = x_t.detach().cpu().numpy().reshape(-1,2)
u = u.detach().cpu().numpy()
fig, ax = plt.subplots()
im = ax.scatter(x_t[:,1], x_t[:,0], c=u[:,0], marker="x", s=10)
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.colorbar(mappable=im)


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




coords = dataset.get_coords().detach().cpu()
data = dataset.get_data().detach().cpu()

fig, ax = plt.subplots()
im = ax.scatter(coords[:,1], coords[:,0], c=data[:,0], marker="x", s=10)
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.colorbar(mappable=im)

plt.show()




train_dataloader, test_dataloader = get_train_test_loader(
    dataset, train_test_split=1.00)


##########################
##########################

poly_order = 2
diff_order = 3

n_combinations = (poly_order+1)*(diff_order+1) 
n_features = 1

network = NN(2, [32, 32, 32, 32], 1)

library = Library1D(poly_order, diff_order)
estimator = Threshold(0.1) 
sparsity_scheduler = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
constraint = LeastSquares()

# Configuration of the sparsity scheduler
model = DeepMoD(network, library, estimator, constraint).to(device)

# Defining optimizer
optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 


foldername = "./data/deepymod/KDV_random/"
#shutil.rmtree(foldername)

#######################################
#######################################
#######################################


# Example usage:
#directory_path = "/path/to/your/directory"
create_or_reset_directory(foldername)


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
    #delta=1e-4,
    #patience=200,
    )



model.sparsity_masks

print(model.sparsity_masks)
print(model.estimator_coeffs())
print(model.constraint.coeff_vectors)


##############################################
##############################################


from deepymod.utils import plot_config_file
from deepymod.analysis import load_tensorboard
history = load_tensorboard(foldername)
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
line_width=2

for history_key in history.keys():
    history_key_parts = history_key.split("_")
      
    if history_key_parts[0] == "estimator":
        if history_key_parts[3] == "0":
            
            if history_key_parts[5] != "3" and history_key_parts[5] != "5":
                
                axs[0].semilogx(
                history[history_key].loc[100:],
                
                linestyle="--",
                linewidth= line_width
                )
        
            elif history_key_parts[5] == "3":
                    
                axs[0].semilogx(
                    history[history_key].loc[100:],
                    label= "\r $u_{xxx}$",
                    
                    linewidth= line_width + 1
                    )
            elif history_key_parts[5] == "5":
            
                axs[0].semilogx(
                    history[history_key].loc[100:],
                    label= "\r $uu_x$",
                    
                    linewidth= line_width + 1
                    ) 
            
            
    
axs[0].set_ylim([-100, 100])
axs[0].set_xlabel("Iterations")
axs[0].set_ylabel("Coefficients")


# Add legends to the right of the subplots
#axs[0].legend(loc='center left', bbox_to_anchor=(0.45, 0.9), ncol=2)
#axs[0].legend(loc='center left', bbox_to_anchor=(0.0, 1.05), ncol=2)
#axs[0].grid(True)
axs[0].grid(True)


coords = dataset.get_coords().cpu()
data = dataset.get_data().cpu()

im = axs[1].scatter(coords[:,1], coords[:,0], c=data[:,0], marker="o", 
                    label=r"Random samples",
                    s=20)
axs[1].set_xlabel(r'$t$')
axs[1].set_ylabel(r'$x$',labelpad=-10)
axs[1].set_ylim([-30.05, 30.05])
#axs[1].legend(loc='center left', bbox_to_anchor=(0.0, 1.05), ncol=1)
#fig.colorbar(mappable=im)


##################################
##################################
##################################

data = scipy.io.loadmat("data/kdv.mat")
#data = np.load("data/kdv.npy",allow_pickle=True)

## preparing and normalizing the input and output data
t_o = 1 * data["t"].flatten()[0:201, None]
x_o = 1 * data["x"].flatten()[:, None]
Exact = np.real(data["usol"])

X, T = np.meshgrid(x_o, t_o, indexing="ij")
coords_org = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
Exact = Exact.flatten()[:, None]


im = axs[2].scatter(coords_org[:,1], coords_org[:,0], c=Exact[:,:], marker="x", s=10)
axs[2].set_xlabel(r'$t$')
axs[2].set_ylabel(r'$x$', labelpad=-10)
axs[2].set_ylim([-30,30])
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
    frameon=True,)

plt.savefig(foldername +
     "KDV" + "coefficients" + "Random_sampling" + f'{num_of_samples}'  
     +"_poly_order_ " + f'{poly_order}'
     +"_diff_order_ " + f'{diff_order}'
     +
     ".png", bbox_inches='tight',
    dpi=600,)
plt.savefig(foldername +
     "KDV" + "coefficients" + "Random_sampling" + f'{num_of_samples}' 
     +"_poly_order_ " + f'{poly_order}'
     +"_diff_order_ " + f'{diff_order}' 
     +".pdf", bbox_inches='tight',
    dpi=600,)

plt.show()

##########################################################
##########################################################
