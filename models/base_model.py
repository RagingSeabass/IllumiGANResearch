from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from . import utils
from collections import OrderedDict
import os
import scipy.io
from data.arw_image import ARW

class BaseModel(ABC):
    
    def __init__(self, manager):

        self.manager = manager
        self.gpus = manager.get_options().get("gpu_ids")
        
        self.is_cuda_ready = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(self.gpus[0])) if self.is_cuda_ready else torch.device('cpu')  # get device name: CPU or GPU
        
        self.cp_dir = manager.get_cp_dir()
        
        # When batch size is 1, we use instance normalization else batch
        if manager.get_hyperparams().get("batch_size") == 1:
            self.norm_layer = 'instance'
        else:
            self.norm_layer = 'batch'
            
        self.optimizers = []
        self.schedulers = []

    def update_learning_rate(self):
        """Update learning rates for all the networks"""
        for sch in self.schedulers:
            sch.step()
        
        self.manager.get_logger('system').info(f"lr update | {self.optimizers[0].param_groups[0]['lr']}")
    

    ##############################
    # METHODS THAT NEEDS TO BE IMPLEMENTED BY ANY MODEL
    ##############################

    @abstractmethod
    def save_networks(self, epochs):
        """Save the network"""
        pass

    @abstractmethod
    def save_visuals(self, img_number, x_path, epoch):
        """Save the network"""
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass"""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights"""
        pass


