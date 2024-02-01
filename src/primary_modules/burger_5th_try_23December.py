#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:15:30 2023

@author: forootani
"""

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
from deepymod.data.burgers import burgers_delta, burgers_delta_org
from deepymod.model.constraint import LeastSquares, Ridge, STRidgeCons
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D
from deepymod.model.sparse_estimators import Threshold, STRidge
from deepymod.training import train
from deepymod.data.DEIM_class import DEIM


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
torch.manual_seed(50)

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

x = torch.linspace(-4, 4, 500)
t = torch.linspace(0.5, 10.0, 500)


#x = torch.tensor(x)
#t = torch.tensor(t)


load_kwargs = {"x": x, "t": t, "v": v, "A": A}
preprocess_kwargs = {"noise_level": 0.00}


#######################
#######################
import scipy.io



def create_data():
    #data = scipy.io.loadmat("deepymod/data/numerical_data/burgers.mat")
    
    x_o = torch.linspace(-8, 8, 100)
    t_o = torch.linspace(0.5, 10.0, 100)
    v = 0.1
    A = 1.0
    
    _ , Exact = burgers_delta_org( x_o, t_o, v, A)
    
     
    
    #data = scipy.io.loadmat("deepymod/data/numerical_data/burgers.mat")
    
    #t_o = data["t"].flatten()[:, None]
    #x_o = data["x"].flatten()[:, None]
    #Exact = np.real(data["usol"])
    
    deim_instance = DEIM(Exact, 2, t_o, x_o, num_basis = 1)
    S_s, T_s, U_s = deim_instance.execute()
    
    coords = torch.from_numpy(np.stack(((T_s, S_s)), axis=-1))
    data = torch.from_numpy( U_s.reshape(-1,1) )
    
    
    #print(type(S_s))
    
    #coords = torch.hstack( (T_s, S_s))
    #data = torch.reshape(-1, 1)
    
    #print("The coodinates have shape {}".format(coords.shape))
    #print("The data has shape {}".format(data.shape))
    return coords, data




coords_2, data_2 = create_data()






########################


"""

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

"""





#########################
#########################
#########################

dataset = Dataset(
    create_data,
    #load_kwargs=load_kwargs,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": 50},
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
    dataset, train_test_split=0.99)








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
model = DeepMoD(network, library, estimator_2, constraint_3, estimator_2).to(device)


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




