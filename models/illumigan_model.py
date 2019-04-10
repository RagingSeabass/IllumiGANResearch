import os
from collections import OrderedDict

import scipy.io
import torch
import torch.optim as optim
from torchsummary import summary
import numpy as np

from data.arw_image import ARW
from models.base_model import BaseModel
from models.nets import GeneratorUNetV1, Discriminator, GAN_loss
from models.utils import get_lr_scheduler, init_network


class IllumiganModel(BaseModel):
    def __init__(self, manager):
        super().__init__(manager)

        self.norm_layer = ""

        # Define generator network
        self.generator_net = GeneratorUNetV1(
            norm_layer=self.norm_layer)

        # Define discriminator network
        self.discriminator_net = Discriminator()

        
        # Define loss function
        self.generator_l1 = torch.nn.L1Loss()

        # Define GAN loss
        self.GAN_loss = GAN_loss(loss='BCE', device=manager.device)

        # Define generator optimzer
        lr = manager.get_hyperparams().get('lr')
        betas = (manager.get_hyperparams().get('b1'),
                 manager.get_hyperparams().get('b2'))

        if manager.is_train:

             # Get optimizer after we init network
            self.generator_opt = torch.optim.Adam(self.generator_net.parameters(),
                                                    lr=lr,
                                                    betas=betas)
            
            self.discriminator_opt = torch.optim.Adam(self.discriminator_net.parameters(),
                                            lr=lr,
                                            betas=betas)

            # We initialize a network to be trained
            if manager.resume_training:

                self.generator_schedular = get_lr_scheduler(self.generator_opt, manager.get_hyperparams())
                self.discriminator_schedular = get_lr_scheduler(self.discriminator_opt, manager.get_hyperparams())

                self.load_network(manager)
                
                if self.is_cuda_ready:
                    self.generator_net.cuda()
                    #self.generator_net = torch.nn.DataParallel(
                    #    self.generator_net).cuda()
                    #self.discriminator_net = torch.nn.DataParallel(
                    #    self.discriminator_net).cuda()

                for state in self.generator_opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
                
                for state in self.discriminator_opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
                

                self.generator_net.to(manager.device)
                self.discriminator_net.to(manager.device)

                self.manager.get_logger('train').info(
                    f"Loaded model at checkpoint {manager.get_hyperparams().get('epoch')}")

            else:

                if self.is_cuda_ready:
                    self.generator_net = torch.nn.DataParallel(
                        self.generator_net).cuda()
                    self.discriminator_net = torch.nn.DataParallel(
                        self.discriminator_net).cuda()

                # Create new model and send it to device
                self.generator_net = init_network(
                    self.generator_net, gpu_ids=self.gpus)
                self.generator_net.to(manager.device)

                self.discriminator_net = init_network(
                    self.discriminator_net, gpu_ids=self.gpus)
                self.discriminator_net.to(manager.device)
                

                self.generator_schedular = get_lr_scheduler(
                    self.generator_opt, manager.get_hyperparams())
                self.discriminator_schedular = get_lr_scheduler(
                    self.discriminator_opt, manager.get_hyperparams())

                self.manager.get_logger('train').info(f"Created new model")

            self.generator_net.train()
            self.discriminator_net.train()

        else:
            
            # Get optimizer after we init network
            self.generator_opt = torch.optim.Adam(self.generator_net.parameters(),
                                                    lr=lr,
                                                    betas=betas)
            
            self.discriminator_opt = torch.optim.Adam(self.discriminator_net.parameters(),
                                            lr=lr,
                                            betas=betas)

            # not used in testing, but needed for load network
            self.generator_schedular = get_lr_scheduler(
                    self.generator_opt, manager.get_hyperparams())
            
            self.discriminator_schedular = get_lr_scheduler(
                    self.discriminator_opt, manager.get_hyperparams())

            self.load_network(manager)

            self.manager.get_logger('test').info(
                f"Loaded model at checkpoint {manager.get_hyperparams().get('epoch')}")


    def load_network(self, manager):

        epochs = manager.get_hyperparams().get("epoch")

        filename_gn = f"{epochs}_generator_net.pth"
        filename_ds = f"{epochs}_discriminator_net.pth"

        load_gn = os.path.join(self.load_dir, filename_gn)
        load_ds = os.path.join(self.load_dir, filename_ds)

        checkpoint_gn = torch.load(load_gn, map_location=manager.device)
        #checkpoint_ds = torch.load(load_ds, map_location=manager.device)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint_gn['net_state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        self.generator_net.load_state_dict(new_state_dict)
        #self.generator_opt.load_state_dict(checkpoint_gn['gen_opt_state_dict'])

        #self.discriminator_net.load_state_dict(checkpoint_ds['disc_state_dict'])
        #self.discriminator_opt.load_state_dict(checkpoint_ds['disc_opt_state_dict'])
        
        #self.generator_schedular.load_state_dict(
        #    checkpoint_gn['schedular_state_dict'])
        #self.discriminator_schedular.load_state_dict(
        #    checkpoint_ds['schedular_state_dict'])

    def save_networks(self, epochs):
        """Save the different models into the same"""
        save_filename_gn = f"{epochs}_generator_net.pth"
        save_filename_ds = f"{epochs}_discriminator_net.pth"
        save_gn = os.path.join(self.save_dir, save_filename_gn)
        save_ds = os.path.join(self.save_dir, save_filename_ds)

        # Load from data parallelize
        if len(self.gpus) > 1:
            generator_net = self.generator_net.module
            discriminator_net = self.discriminator_net.module
        else:
            generator_net = self.generator_net
            discriminator_net = self.discriminator_net

        torch.save({
            'gen_state_dict': generator_net.state_dict(),
            'gen_opt_state_dict': self.generator_opt.state_dict(),
            'schedular_state_dict': self.generator_schedular.state_dict()
        }, save_gn)

        torch.save({
            'disc_state_dict': discriminator_net.state_dict(),
            'disc_opt_state_dict': self.discriminator_opt.state_dict(),
            'schedular_state_dict': self.discriminator_schedular.state_dict()
        }, save_ds)

        # Move models back to gpu after save
        if self.is_cuda_ready:
            self.generator_net.cuda()
            self.discriminator_net.cuda()

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

    def set_input(self, x, x_processed, y):
        """Takes input of form X Y and sends it to the GPU"""

        x = x.permute(0, 3, 1, 2).to(self.manager.device)
        x_processed = x_processed.permute(0, 3, 1, 2).to(self.manager.device)
        y = y.permute(0, 3, 1, 2).to(self.manager.device)

        self.x = x
        self.x_processed = x_processed
        self.y = y

    def forward(self):
        """Make generator"""
        self.fake_y = self.generator_net(self.x)  # G(X) = fake_y

    def g_backward(self):
        # GAN Loss
        
        fake_pair = torch.cat((self.x_processed, self.fake_y), 1)
        fake_prediction = self.discriminator_net(fake_pair)
        self.GAN_loss_generator = self.GAN_loss.compute(fake_prediction, 1)

        # L1 Loss
        self.generator_l1_loss = self.generator_l1(self.fake_y, self.y) * 100
        # Overall loss of generator_net
        self.generator_loss = self.GAN_loss_generator + self.generator_l1_loss
        # Compute gradients
        self.generator_loss.backward()

    def d_backward(self):
        # Calculate loss on pair of real images
        real_pair = torch.cat((self.x_processed, self.y), 1)
        real_prediction = self.discriminator_net(real_pair)
        real_loss = self.GAN_loss.compute(real_prediction, 1.0)

        # Calculate loss on pair of real input and fake output image
        fake_pair = torch.cat((self.x_processed, self.fake_y), 1)
        # Detatch to prevent backprop on generator_net
        fake_pair = fake_pair.detach()
        fake_prediction = self.discriminator_net(fake_pair)
        fake_loss = self.GAN_loss.compute(fake_prediction, 0.0)

        # Overall loss of discriminator_net
        self.discriminator_loss = (real_loss + fake_loss) * 0.5
        # Compute gradients
        self.discriminator_loss.backward()

    def get_L1_loss(self):
        return self.generator_l1_loss.item()


    def get_generator_loss(self):
        return self.generator_loss.item()

    def get_discriminator_loss(self):
        return self.discriminator_loss.item()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights"""
        
        # Calc G(x) (fake_y)
        self.forward()
        # Train Discriminator

        # Allow backpropogation of discriminator
        for param in self.discriminator_net.parameters():
            param.requires_grad = True

        # Set D's gradients to zero
        self.discriminator_opt.zero_grad()
        
        # Backpropagate
        self.d_backward()
        # Update weights
        self.discriminator_opt.step()
        # Disable backpropogation of discriminator when training G
        for param in self.discriminator_net.parameters():
            param.requires_grad = False

        # Train Generator

        # set G's gradients to zero
        self.generator_opt.zero_grad()
        # backpropagate
        self.g_backward()
        # Update weights
        self.generator_opt.step()

    def update_lr(self, epoch):

        if epoch > self.manager.get_hyperparams().get('epoch_iterations'):
            for g in self.generator_opt.param_groups:
                g['lr'] = 1e-5
            for g in self.discriminator_opt.param_groups:
                g['lr'] = 1e-5

        self.manager.get_logger('system').info(
            f"lr update | {self.generator_opt.param_groups[0]['lr']}")

    def update_learning_rate(self):
        """Update learning rate"""
        self.generator_schedular.step()
        self.discriminator_schedular.step()
        
        self.manager.get_logger('system').info(
            f"lr update | {self.generator_opt.param_groups[0]['lr']}")

    def test(self):
        with torch.no_grad():  # disable back prop
            self.forward()
            self.generator_l1_loss = self.generator_l1(self.fake_y, self.y)
