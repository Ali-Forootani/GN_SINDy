
import sys
import os
import scipy.io as sio

cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
import torch
import matplotlib.pylab as plt
from scipy.io import loadmat
#from Functions.base import Dataset, get_train_test_loader, scale_to_range
#from DEIM_3D_class import DEIM

##############################
##############################

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

data = loadmat("data/reaction_diffusion_standard.mat")