from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from . import utils
from collections import OrderedDict
import os

class BaseModel(ABC):
    
    def __init__(self, manager):

        self.manager = manager
        self.gpus = manager.get_options().get("gpu_ids")
        
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(self.gpus[0])) if self.cuda else torch.device('cpu')  # get device name: CPU or GPU
        
        self.cp_dir = manager.get_cp_dir()
        
        # When batch size is 1, we use instance normalization else batch
        if manager.get_hyperparams().get("batch_size") == 1:
            self.norm_layer = 'instance'
        else:
            self.norm_layer = 'batch'

        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        

    def update_learning_rate(self):
        """Update learning rates for all the networks"""
        for sch in self.schedulers:
            sch.step()
        
        if self.manager.is_train:
            self.manager.get_logger('system').info(f"lr update | {self.optimizers[0].param_groups[0]['lr']}")
    

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk"""
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.cp_dir, save_filename)
                net = getattr(self, name + '_net')

                if len(self.gpus) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpus[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    @abstractmethod
    def forward(self):
        """Run forward pass"""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights"""
        pass


