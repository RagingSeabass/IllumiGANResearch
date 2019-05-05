import os
from abc import ABC, abstractmethod
from collections import OrderedDict

import scipy.io
import torch
import torch.nn as nn

from data.arw_image import ARW

from . import utils


class BaseModel(ABC):
    
    def __init__(self, manager):

        self.manager = manager
        self.gpus = manager.get_options().get("gpu_ids")
        
        self.is_cuda_ready = torch.cuda.is_available()

        self.save_dir = manager.get_save_dir()
        self.load_dir = manager.get_load_dir()
                
        # When batch size is 1, we use instance normalization else batch
        if manager.get_hyperparams().get("batch_size") == 1:
            self.norm_layer = 'instance'
        else:
            self.norm_layer = 'batch'
  
    ##############################
    # METHODS THAT NEEDS TO BE IMPLEMENTED BY ANY MODEL
    ##############################

    @abstractmethod
    def save_networks(self, epochs):
        """Save the network"""
        pass

    @abstractmethod
    def save_visuals(self, img_number, epoch):
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
