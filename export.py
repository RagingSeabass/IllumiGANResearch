import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.arw_dataset import ARWDataset
from models.illumigan_model import IllumiganModel
from utils import Average, TrainManager
import torch.backends.cudnn as cudnn

base_dir = "_default/"
server = False

if len(sys.argv) > 1:
    base_dir = str(sys.argv[1])

if len(sys.argv) > 2:
    base_dir = str(sys.argv[1])
    server = True

# Temporary defined options

cudnn.benchmark = True

# ------- 

options = './experiments/base_model/local_options.json'
hyperparams = './experiments/base_model/local_params.json'

if server:
    options = './experiments/base_model/options.json'
    hyperparams = './experiments/base_model/params.json'
    

manager = TrainManager(base_dir=base_dir,
                       options_f_dir=options,
                       hyperparams_f_dir=hyperparams)

dataset = ARWDataset(manager, 'short', 'long')
dataloader = DataLoader(dataset, batch_size=manager.get_hyperparams().get(
    'batch_size'), shuffle=True, num_workers=0)

model = IllumiganModel(manager=manager)



for i, (x, x_processed, y) in enumerate(dataloader):
    
    model.set_input(x, x_processed, y)
    

    torch.onnx.export(model.generator_net, x, "test.onnx")
   
