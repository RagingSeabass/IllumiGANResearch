import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.arw_dataset import ARWDataset
from data.png_dataset import PNGDataset
from models.illumigan_model import IllumiganModel
from utils import Average, TestManager

base_dir = "_default"

server = False

if len(sys.argv) > 1:
    base_dir = str(sys.argv[1])

if len(sys.argv) > 2:
    base_dir = str(sys.argv[1])
    server = True

options = './experiments/base_model/local_options.json'
hyperparams = './experiments/base_model/local_params.json'

if server:
    options = './experiments/base_model/test_options.json'
    hyperparams = './experiments/base_model/test_params.json'
    


manager = TestManager(base_dir=base_dir,
                      options_f_dir=options,
                      hyperparams_f_dir=hyperparams)

#dataset = ARWDataset(manager, 'short', 'long')
dataset = PNGDataset(manager, 'in', 'out', transforms=True)
# We only allow testing on batch size 1
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

model = IllumiganModel(manager=manager)

total_iterations = 0    # total iterations
loss = Average()

train_start_time = time.time()  # timer for entire epoch
loss.reset()

for i, (x, y) in enumerate(dataloader):

    total_iterations += 1

    # Get the only element in the batch

    model.set_input(x, x, y)
    model.test()

    loss.update(model.get_L1_loss())

    # Save previes of model images
    if manager.options.get("save_images"):
        model.save_visuals(i, 'test')

    manager.get_logger("test").info(
        f"Image {i} | Loss {model.get_L1_loss()} | Time {time.time() - train_start_time} | Iteration {total_iterations}")

manager.get_logger("test").info(
    f"Average Loss {loss.average()} | Time {time.time() - train_start_time} | Iteration {total_iterations}")
