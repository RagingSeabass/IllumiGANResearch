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

    def save_visuals(self, img_number, x_path, epoch):
        
        if not os.path.isdir(self.manager.get_img_dir() + str(epoch) + '/'):
            os.makedirs(self.manager.get_img_dir() + str(epoch) + '/')

        for i, y in enumerate(self.y):
            x = x_path[i]
            real_y_rgb = y.cpu().data.numpy()               # 3, 1024, 1024
            fake_y_rgb = self.fake_y[i].cpu().data.numpy()  # 3, 1024, 1024

            arw = ARW(x)
            arw.postprocess()

            scipy.misc.toimage(arw.get() * 255, high=255, low=0, cmin=0, cmax=255).save(
            self.manager.get_img_dir() + f"{epoch}/{img_number}_{i}_x.png")
            scipy.misc.toimage(real_y_rgb * 255, high=255, low=0, cmin=0, cmax=255).save(
            self.manager.get_img_dir() + f"{epoch}/{img_number}_{i}_y.png")
            scipy.misc.toimage(fake_y_rgb * 255, high=255, low=0, cmin=0, cmax=255).save(
            self.manager.get_img_dir() + f"{epoch}/{img_number}_{i}_y_pred.png")
            

    @abstractmethod
    def forward(self):
        """Run forward pass"""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights"""
        pass


