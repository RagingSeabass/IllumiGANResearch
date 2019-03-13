from utils import *
from data.arw_dataset import ARWDataset
import os
import rawpy
import numpy as np

m = TrainManager(base_dir='test', 
                options_f_dir='./experiments/base_model/options.json',  
                hyperparams_f_dir='./experiments/base_model/params.json')

data = ARWDataset(m, 'short', 'long')

for i in range(len(data)):
    sample = data[i]

