import torch
from models.illumigan_model import IllumiganModel
from utils import Average, TrainManager
import os
import sys
import time


base_dir = "_default/"

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


model = IllumiganModel(manager=manager)

# Create the right input shape (e.g. for an image)
dummy_input = torch.randn(1, 4, 2832, 4240)

torch.onnx.export(model.generator_net, dummy_input, "Illumigan.onnx")