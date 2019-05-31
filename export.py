import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.png_dataset import PNGDataset
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

dataset = PNGDataset(manager, 'in', 'out', transforms=True)
dataloader = DataLoader(dataset, batch_size=manager.get_hyperparams().get(
    'batch_size'), shuffle=True, num_workers=0)

model = IllumiganModel(manager=manager)

dummy_input = torch.randn(1, 3, 512, 512).to(manager.device)

torch.onnx.export(model.generator_net, dummy_input, "Illumigan.onnx")

onnx_model = onnx.load('./Illumigan.onnx')


# If we normalize between -1 and 1 
# use this scale 
#scale = 2/255.0
#args = dict(
#    is_bgr=False,
#    red_bias = -1,
#    green_bias = -1, 
#    blue_bias = -1,
#    image_scale = scale
#)

# IF we normalize between 0 and 1 and input
# use this scale
scale = 1/255.0
args = dict(
    is_bgr=False,
    red_bias = 0,
    green_bias = 0, 
    blue_bias = 0,
    image_scale = scale
)
# This is what makes it an image class
mlmodel = convert(onnx_model, image_input_names='0', preprocessing_args=args) 
mlmodel.save('Illumigan.mlmodel')