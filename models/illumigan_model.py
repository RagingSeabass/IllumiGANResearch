import os
from collections import OrderedDict

import scipy.io
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchsummary import summary

from data.arw_image import ARW
from models.base_model import BaseModel
from models.nets import GeneratorUNetV1
from models.utils import get_lr_scheduler, init_network


class IllumiganModel(BaseModel):
    def __init__(self, manager):
        super().__init__(manager)

        cudnn.benchmark = True

        self.generator_net = GeneratorUNetV1(
            norm_layer=self.norm_layer, use_dropout=False)
        self.generator_opt = torch.optim.Adam(
            self.generator_net.parameters(),
            lr=manager.get_hyperparams().get('lr'),
            betas=(0.5, 0.999))
        self.generator_l1 = torch.nn.L1Loss()

        if manager.is_train:
            
            # We initialize a network to be trained
            if manager.get_hyperparams().get("epoch") > 0:
                
                epoch = manager.get_hyperparams().get("epoch")
                self.load_network(epoch)
        
                self.manager.get_logger('train').info(f"Loaded model at checkpoint {epoch}")
            
            self.generator_net = init_network(self.generator_net, gpu_ids=self.gpus)
            self.optimizers.append(self.generator_opt)
            self.schedulers = [get_lr_scheduler(
                optimizer, manager.get_hyperparams()) for optimizer in self.optimizers]
            self.generator_net.train()

        else:

            epoch = manager.get_hyperparams().get("epoch")
            self.generator_net = init_network(self.generator_net, gpu_ids=self.gpus)
            self.load_network(epoch)

            self.manager.get_logger('test').info(f"Loaded model at checkpoint {epoch}")

        summary(self.generator_net, input_size=(4, 512, 512))
        
    def load_network(self, epochs):
        
        filename_gn = f"{epochs}_generator_net.pth"
        filename_go = f"{epochs}_generator_opt.pth"

        load_gn = os.path.join(self.load_dir, filename_gn)
        load_go = os.path.join(self.load_dir, filename_go)

        generator_net_checkpoint = torch.load(load_gn)
        generator_opt_checkpoint = torch.load(load_go)

        # load params

        self.generator_net.load_state_dict(
                    generator_net_checkpoint['generator_net_state_dict'])
        self.generator_opt.load_state_dict(
                    generator_opt_checkpoint['generator_opt_state_dict'])

        # Move models back to gpu after save
        if len(self.gpus) > 0:
            if self.is_cuda_ready:
                self.generator_opt.cuda()
                self.generator_net.to(self.gpus[0])
            self.generator_net = torch.nn.DataParallel(self.generator_net, self.gpus)  # multi-GPUs

    def save_networks(self, epochs):
        """Save the different models into the same"""
        save_filename_gn = f"{epochs}_generator_net.pth"
        save_filename_go = f"{epochs}_generator_opt.pth"
        save_gn = os.path.join(self.save_dir, save_filename_gn)
        save_go = os.path.join(self.save_dir, save_filename_go)

        # Load from data parallelize
        if len(self.gpus) > 0 and self.is_cuda_ready:
            generator_net = self.generator_net.module.cpu()
        else:
            generator_net = self.generator_net.cpu()
        
        generator_opt = self.generator_opt

        torch.save({
            'generator_net_state_dict': generator_net.state_dict(),
        }, save_gn)

        torch.save({
            'generator_opt_state_dict': generator_opt.state_dict(),
        }, save_go)

        # Move models back to gpu after save
        if self.is_cuda_ready:
            self.generator_net.cuda()

    def save_visuals(self, num, x_path, epoch):
        if not os.path.isdir(self.manager.get_img_dir() + str(epoch) + '/'):
            os.makedirs(self.manager.get_img_dir() + str(epoch) + '/')

        for i, y in enumerate(self.y):
            # path to original img
            x = x_path[i]
            real_y_rgb = y.cpu().data.numpy()               # 3, 1024, 1024 (Crop)
            # 3, 1024, 1024 (Crop)
            fake_y_rgb = self.fake_y[i].cpu().data.numpy()

            arw = ARW(x)
            arw.postprocess()

            scipy.misc.toimage(arw.get() * 255, high=255, low=0, cmin=0, cmax=255).save(
                self.manager.get_img_dir() + f"{epoch}/{num}_{i}_x.png")
            scipy.misc.toimage(real_y_rgb * 255, high=255, low=0, cmin=0, cmax=255).save(
                self.manager.get_img_dir() + f"{epoch}/{num}_{i}_y.png")
            scipy.misc.toimage(fake_y_rgb * 255, high=255, low=0, cmin=0, cmax=255).save(
                self.manager.get_img_dir() + f"{epoch}/{num}_{i}_y_pred.png")

    def set_input(self, x, y):
        """Takes input of form X Y and sends it to the GPU"""

        x = x.permute(0, 3, 1, 2).to(self.device)
        y = y.permute(0, 3, 1, 2).to(self.device)

        self.x = x
        self.y = y

    def forward(self):
        """Make generator"""
        self.fake_y = self.generator_net(self.x)  # G(X) = fake_y

    def g_backward(self):
        self.generator_l1_loss = self.generator_l1(self.fake_y, self.y)
        self.generator_l1_loss.backward()

    def get_L1_loss(self):
        return self.generator_l1_loss.item()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights"""
        # Calc G(x)
        self.forward()

        # set G's gradients to zero
        self.generator_opt.zero_grad()

        # back propagate
        self.g_backward()

        # Update weights
        self.generator_opt.step()

    def test(self):
        with torch.no_grad():  # disable back prop
            self.forward()
            self.generator_l1_loss = self.generator_l1(self.fake_y, self.y)
