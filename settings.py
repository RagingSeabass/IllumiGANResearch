from utils import *
from data.arw_dataset import ARWDataset
import os
import rawpy
import numpy as np

m = TrainManager(base_dir='test', param_dir='./experiments/base_model/params.json')

data = ARWDataset(m.get_params(), m.get_data_dir(), 'short', 'long')

for i in range(len(data)):
    sample = data[i]

