import os
from collections import OrderedDict

import scipy.io
import torch
import torch.optim as optim
from torchsummary import summary
import numpy as np

from data.arw_image import ARW
from models.base_model import BaseModel
from models.nets import GeneratorUNetV1
from models.utils import get_lr_scheduler, init_network


class IllumiganModel(BaseModel):
    def __init__(self, manager):
        super().__init__(manager)


        self.norm_layer = ""

        # Define generator network
        self.generator_net = GeneratorUNetV1(
            norm_layer=self.norm_layer, use_dropout=False)

        if self.is_cuda_ready:
            self.generator_net = torch.nn.DataParallel(
                self.generator_net).cuda()

        # Define loss function
        self.generator_l1 = torch.nn.L1Loss()

        # Define generator optimzer
        lr = manager.get_hyperparams().get('lr')
        betas = (manager.get_hyperparams().get('b1'),
                 manager.get_hyperparams().get('b2'))

        self.generator_opt = torch.optim.Adam(self.generator_net.parameters(),
                                              lr=lr,
                                              betas=betas)

        

        if manager.is_train:

            # We initialize a network to be trained
            if manager.resume_training:

                self.generator_schedular = get_lr_scheduler(
                    self.generator_opt, manager.get_hyperparams())

                self.load_network(manager)

                self.manager.get_logger('train').info(
                    f"Loaded model at checkpoint {manager.get_hyperparams().get('epoch')}")

            else:

                # Create new model and send it to device
                self.generator_net = init_network(
                    self.generator_net, gpu_ids=self.gpus)
                self.generator_net.to(manager.device)

                # Define generator optimzer
                lr = manager.get_hyperparams().get('lr')
                betas = (manager.get_hyperparams().get('b1'),
                         manager.get_hyperparams().get('b2'))

                self.generator_opt = torch.optim.Adam(self.generator_net.parameters(),
                                                      lr=lr,
                                                      betas=betas)

                self.generator_schedular = get_lr_scheduler(
                    self.generator_opt, manager.get_hyperparams())

                self.manager.get_logger('train').info(f"Created new model")

            self.optimizers.append(self.generator_opt)
            self.generator_net.train()

        else:
            # not used in testing, but needed for load network
            self.generator_schedular = get_lr_scheduler(
                    self.generator_opt, manager.get_hyperparams())
            self.load_network(manager)

            self.manager.get_logger('test').info(
                f"Loaded model at checkpoint {manager.get_hyperparams().get('epoch')}")

        summary(self.generator_net, input_size=(4, 512, 512))

    def load_network(self, manager):

        epochs = manager.get_hyperparams().get("epoch")

        filename_gn = f"{epochs}_generator_net.pth"

        load_gn = os.path.join(self.load_dir, filename_gn)

        checkpoint = torch.load(load_gn, map_location=manager.device)

        self.generator_net.load_state_dict(checkpoint['net_state_dict'])
        self.generator_opt.load_state_dict(checkpoint['opt_state_dict'])
        self.generator_schedular.load_state_dict(
            checkpoint['schedular_state_dict'])

        # for state in self.generator_opt.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.to(manager.device)

    def save_networks(self, epochs):
        """Save the different models into the same"""
        save_filename_gn = f"{epochs}_generator_net.pth"
        save_gn = os.path.join(self.save_dir, save_filename_gn)

        # Load from data parallelize
        if len(self.gpus) > 1:
            generator_net = self.generator_net.module
        else:
            generator_net = self.generator_net

        torch.save({
            'net_state_dict': generator_net.state_dict(),
            'opt_state_dict': self.generator_opt.state_dict(),
            'schedular_state_dict': self.generator_schedular.state_dict()
        }, save_gn)

        # Move models back to gpu after save
        if self.is_cuda_ready:
            self.generator_net.cuda()

    def save_visuals(self, num, epoch):
        if not os.path.isdir(self.manager.get_img_dir() + str(epoch) + '/'):
            os.makedirs(self.manager.get_img_dir() + str(epoch) + '/')

        for i, y in enumerate(self.y):
            # path to original img
            # 3, 1024, 1024 (Crop)
            real_y_rgb = y.data.cpu().numpy()               

            # 3, 1024, 1024 (Crop)
            fake_y_rgb = self.fake_y[i].data.cpu().numpy()

            real_y_rgb = real_y_rgb.transpose(1, 2, 0) * 225
            fake_y_rgb = fake_y_rgb.transpose(1, 2, 0) * 225

            temp = np.concatenate(
                (real_y_rgb[:, :, :], fake_y_rgb[:, :, :]), axis=1)
            scipy.misc.toimage(temp, high=255, low=0, cmin=0, cmax=255).save(
                self.manager.get_img_dir() + f"{epoch}/{num}_{i}.png")

    def set_input(self, x, y):
        """Takes input of form X Y and sends it to the GPU"""

        x = x.permute(0, 3, 1, 2).to(self.manager.device)
        y = y.permute(0, 3, 1, 2).to(self.manager.device)

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

    def update_lr(self, epoch):

        if epoch > self.manager.get_hyperparams().get('epoch_iterations'):
            for g in self.generator_opt.param_groups:
                g['lr'] = 1e-5

        self.manager.get_logger('system').info(
            f"lr update | {self.generator_opt.param_groups[0]['lr']}")

    def update_learning_rate(self):
        """Update learning rate"""
        self.generator_schedular.step()
        self.manager.get_logger('system').info(
            f"lr update | {self.generator_opt.param_groups[0]['lr']}")

    def test(self):
        with torch.no_grad():  # disable back prop
            self.forward()
            self.generator_l1_loss = self.generator_l1(self.fake_y, self.y)
