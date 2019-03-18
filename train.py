from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import os, time, sys


from utils import TrainManager
from data.arw_dataset import ARWDataset
from models.illumigan_model import IllumiganModel
from utils import Average

base_dir = "_default"

if len(sys.argv) > 1:
        base_dir = str(sys.argv[1])

manager = TrainManager(base_dir=base_dir, 
                options_f_dir='./experiments/base_model/local_options.json',  
                hyperparams_f_dir='./experiments/base_model/params.json')

dataset = ARWDataset(manager, 'short', 'long')

# TODO: CHECK IF THE NUM WORKERS CAN BE IMPLEMENTED. IT STALLS THE DATA LOADER FOR SOME REASON

dataloader = DataLoader(dataset, batch_size=manager.get_hyperparams().get('batch_size'), shuffle=True, num_workers=0)

model = IllumiganModel(manager=manager)

total_iterations = 0    # total iterations
epoch_loss = Average()

for epoch in range(manager.get_hyperparams().get('epoch'),              # Starting epoch
    manager.get_hyperparams().get('epoch_iterations') +                 # epochs without decaying lr
    manager.get_hyperparams().get('epoch_decaying_iterations') + 1):    # epochs with decaying lr
    
    epoch_start_time = time.time()  # timer for entire epoch
    iterations = 0                  # the number of training iterations in current epoch

    epoch_loss.reset()

    for x,y in dataloader:

        data_start_time = time.time()
        t_data = data_start_time - epoch_start_time

        total_iterations    += manager.get_hyperparams().get('batch_size')
        iterations          += manager.get_hyperparams().get('batch_size')

        # Get the only element in the batch

        model.set_input(x,y)
        model.optimize_parameters()

        epoch_loss.update(model.get_loss())

    manager.get_logger("train").info(f"Epoch {epoch} | Loss {epoch_loss.average()} | Time {time.time() - epoch_start_time}")

    if epoch % manager.options.get("save") == 0:              # cache our model every <save_epoch_freq> epochs
            manager.get_logger("system").info(f"Saving model at end of Epoch {epoch} | Iteration {iterations}")
            model.save_networks('latest')
            model.save_networks(epoch)        

    model.update_learning_rate()
    


