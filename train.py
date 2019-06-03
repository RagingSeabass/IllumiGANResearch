import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.arw_dataset import ARWDataset
from data.png_dataset import PNGDataset
from models.illumigan_model import IllumiganModel
from utils import Average, TrainManager

base_dir = "_default/"
server = False
cudnn.enabled = True
cudnn.benchmark = True
if len(sys.argv) > 1:
    base_dir = str(sys.argv[1])

if len(sys.argv) > 2:
    base_dir = str(sys.argv[1])
    server = True


# Temporary defined options

# ------- 

options = './experiments/base_model/local_options.json'
hyperparams = './experiments/base_model/local_params.json'

if server:
    options = './experiments/base_model/options.json'
    hyperparams = './experiments/base_model/params.json'
    

manager = TrainManager(base_dir=base_dir,
                       options_f_dir=options,
                       hyperparams_f_dir=hyperparams)

manager.get_logger("system").info(f"Started loading data")

#dataset = JPGDataset(manager, 'short', 'long', transforms=True)
dataset = ARWDataset(manager, 'short', 'long')
#dataset = PNGDataset(manager, 'in', 'out', transforms=True)
dataloader = DataLoader(dataset, batch_size=manager.get_hyperparams().get('batch_size'), shuffle=True, num_workers=0)

model = IllumiganModel(manager=manager)

total_iterations = 0    # total iterations
epoch_loss_generator = Average()
epoch_loss_discriminator = Average()

manager.get_logger("train").info(f"Started training | Iteration {total_iterations}")


for epoch in range(manager.get_hyperparams().get('epoch'),              # Starting epoch
                   # epochs without decaying lr
                   manager.get_hyperparams().get('epoch_iterations') +
                   manager.get_hyperparams().get('epoch_decaying_iterations') + 1):    # epochs with decaying lr

    # update lr
    model.update_learning_rate()

    epoch_start_time = time.time()  # timer for entire epoch

    epoch_loss_generator.reset()
    epoch_loss_discriminator.reset()
    
    for i, (x, x_processed, y) in enumerate(dataloader):
        
        data_start_time = time.time()
        t_data = data_start_time - epoch_start_time

        total_iterations += manager.get_hyperparams().get('batch_size')
        
        # Get the only element in the batch
        model.set_input(x, x_processed, y)
        
        model.optimize_parameters()

        epoch_loss_generator.update(model.get_generator_loss())
        epoch_loss_discriminator.update(model.get_discriminator_loss())

        # Save previes of model images
    
        if manager.options.get("save_images") and epoch % manager.options.get("save_image_epoch") == 0:
            model.save_visuals(i, epoch)

    #manager.get_logger("train").info(
    #    f"Epoch {epoch} | Loss G: {epoch_loss_generator.average()} | Time {time.time() - epoch_start_time} | Iteration {total_iterations}")

    manager.get_logger("train").info(
        f"Epoch {epoch} | Loss G: {epoch_loss_generator.average()} D: { epoch_loss_discriminator.average()} | Time {time.time() - epoch_start_time} | Iteration {total_iterations}")

    # cache our model every <save_epoch_freq> epochs
    if epoch % manager.options.get("save_epoch") == 0:
        manager.get_logger("system").info(
            f"Saved model for Epoch {epoch} | Iteration {total_iterations}")

        if manager.options.get("save_images") and epoch % manager.options.get("save_image_epoch") == 0:
            manager.get_logger("system").info(
                f"Saved images for Epoch {epoch} | Iteration {total_iterations}")

        model.save_networks('latest')
        model.save_networks(epoch)

