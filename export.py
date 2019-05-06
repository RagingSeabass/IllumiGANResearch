import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.jpg_dataset import JPGDataset
from models.illumigan_model import IllumiganModel
from utils import Average, TrainManager
import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = True


import onnx;
from onnx_coreml import convert

base_dir = "_default/"
server = False

if len(sys.argv) > 1:
    base_dir = str(sys.argv[1])

if len(sys.argv) > 2:
    base_dir = str(sys.argv[1])
    server = True

# ------- 

options = './experiments/base_model/local_options.json'
hyperparams = './experiments/base_model/local_params.json'

if server:
    options = './experiments/base_model/options.json'
    hyperparams = './experiments/base_model/params.json'
    
# working code
manager = TrainManager(base_dir=base_dir,
                       options_f_dir=options,
                       hyperparams_f_dir=hyperparams)

dataset = JPGDataset(manager, 'short', 'long', transforms=True)
dataloader = DataLoader(dataset, batch_size=manager.get_hyperparams().get(
    'batch_size'), shuffle=True, num_workers=0)

model = IllumiganModel(manager=manager)


dummy_input = torch.randn(1, 3, 256, 256).to(manager.device)

torch.onnx.export(model.generator_net, dummy_input, "Illumigan.onnx")

onnx_model = onnx.load('./Illumigan.onnx')
mlmodel = convert(onnx_model)
mlmodel.save('coreml_model.mlmodel')