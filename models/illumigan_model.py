import os
from collections import OrderedDict

import scipy.io
import torch
import torch.optim as optim
from torchsummary import summary
import numpy as np

from PIL import Image
from data.arw_image import ARW
from models.base_model import BaseModel
from models.nets import GeneratorUNetV1, Discriminator, GAN_loss, PatchDiscriminator
from models.utils import get_lr_scheduler, init_network

from .utils import tensor2img

class IllumiganModel(BaseModel):
    def __init__(self, manager):
        super().__init__(manager)

        # Define generator network
        self.generator_net = GeneratorUNetV1(norm_layer=self.norm_layer)

        # Define discriminator network

        if manager.get_hyperparams().get("dis") == 'patch':
            self.discriminator_net = PatchDiscriminator(input_nc=6, ndf=64, n_layers=3)
        else:
            discriminator_net = Discriminator()
    
        # Define loss function
        self.generator_l1 = torch.nn.L1Loss()

        # Define GAN loss
        self.GAN_loss = GAN_loss(loss='BCEWithLogitsLoss', device=manager.device)

        # Define generator optimzer
        lr = manager.get_hyperparams().get('lr')
        lr_dis = manager.get_hyperparams().get('lr')
        
        betas = (manager.get_hyperparams().get('b1'),
                 manager.get_hyperparams().get('b2'))


        if manager.is_train:            
            ###
            # NETWORK IS GETTING TRAINED
            ###

            # We initialize a network to be trained
            if manager.resume_training:
                
                # 
                self.generator_opt = torch.optim.Adam(self.generator_net.parameters(),
                                                        lr=lr,
                                                        betas=betas)
                
                self.discriminator_opt = torch.optim.Adam(self.discriminator_net.parameters(),
                                                lr=lr_dis,
                                                betas=betas)

                ###
                # Continue training
                ###

                self.generator_schedular = get_lr_scheduler(self.generator_opt, manager.get_hyperparams())
                self.discriminator_schedular = get_lr_scheduler(self.discriminator_opt, manager.get_hyperparams())

                self.load_network(manager)

                # Move everything to the GPU
                # Test this
                for state in self.generator_opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(manager.device)
                
                for state in self.discriminator_opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(manager.device)


                self.manager.get_logger('train').info(
                    f"Loaded model at checkpoint {manager.get_hyperparams().get('epoch')}")

            else:

                ###
                # New training
                ###

                # Create new model and send it to device
                self.generator_net = init_network(self.generator_net)
                self.generator_net.to(manager.device)

                self.discriminator_net = init_network(self.discriminator_net)
                self.discriminator_net.to(manager.device)
                
                # Get optimizer after we init network
                self.generator_opt = torch.optim.Adam(self.generator_net.parameters(),
                                                        lr=lr,
                                                        betas=betas)
                
                self.discriminator_opt = torch.optim.Adam(self.discriminator_net.parameters(),
                                                lr=lr_dis,
                                                betas=betas)


                self.generator_schedular = get_lr_scheduler(
                    self.generator_opt, manager.get_hyperparams())
                self.discriminator_schedular = get_lr_scheduler(
                    self.discriminator_opt, manager.get_hyperparams())

                self.manager.get_logger('train').info(f"Created new model")

        else:
            
            ###
            # Testing model
            ###

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
        checkpoint_ds = torch.load(load_ds, map_location=manager.device)

        self.generator_net.load_state_dict(checkpoint_gn['gen_state_dict'])
        self.generator_opt.load_state_dict(checkpoint_gn['gen_opt_state_dict'])

        self.discriminator_net.load_state_dict(checkpoint_ds['disc_state_dict'])
        self.discriminator_opt.load_state_dict(checkpoint_ds['disc_opt_state_dict'])
        
        self.generator_schedular.load_state_dict(
            checkpoint_gn['schedular_state_dict'])
        self.discriminator_schedular.load_state_dict(
            checkpoint_ds['schedular_state_dict'])
                    
        self.generator_net.to(manager.device)
        self.discriminator_net.to(manager.device)

    def save_networks(self, epochs):
        """Save the different models into the same"""
        save_filename_gn = f"{epochs}_generator_net.pth"
        save_filename_ds = f"{epochs}_discriminator_net.pth"
        save_gn = os.path.join(self.save_dir, save_filename_gn)
        save_ds = os.path.join(self.save_dir, save_filename_ds)

        torch.save({
            'gen_state_dict': self.generator_net.state_dict(),
            'gen_opt_state_dict': self.generator_opt.state_dict(),
            'schedular_state_dict': self.generator_schedular.state_dict()
        }, save_gn)

        torch.save({
           'disc_state_dict': self.discriminator_net.state_dict(),
           'disc_opt_state_dict': self.discriminator_opt.state_dict(),
           'schedular_state_dict': self.discriminator_schedular.state_dict()
        }, save_ds)

        # Move models back to gpu after save
        
        self.generator_net.to(self.manager.device)
        self.discriminator_net.to(self.manager.device)

    def save_visuals(self, num, epoch):
        
        # This needs to be tested!!! 

        if not os.path.isdir(self.manager.get_img_dir() + str(epoch) + '/'):
            os.makedirs(self.manager.get_img_dir() + str(epoch) + '/')

        for i, y in enumerate(self.y):
            # path to original img
            # 3, 1024, 1024 (Crop)
            real_y_rgb = tensor2img(y)            

            # 3, 1024, 1024 (Crop)
            fake_y_rgb = tensor2img(self.fake_y[i])
            
            temp = np.concatenate(
                (real_y_rgb[:, :, :], fake_y_rgb[:, :, :]), axis=1)
            
            image_pil = Image.fromarray(temp)
            image_pil.save(self.manager.get_img_dir() + f"{epoch}/{num}_{i}.png")


    def set_input(self, x, x_processed, y):
        """Takes input of form X Y and sends it to the GPU"""

        self.x = x.to(self.manager.device)
        self.x_processed =  x_processed.to(self.manager.device)
        self.y = y.to(self.manager.device)

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
        self.generator_loss = self.generator_l1_loss  + self.GAN_loss_generator
        
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
            f"lr | {self.generator_opt.param_groups[0]['lr']}")

    def update_learning_rate(self):
        """Update learning rate"""
        self.generator_schedular.step()
        self.discriminator_schedular.step()
        
        self.manager.get_logger('system').info(
            f"lr | {self.generator_opt.param_groups[0]['lr']}")

    def test(self):
        with torch.no_grad():  # disable back prop
            self.forward()
            #self.generator_l1_loss = self.generator_l1(self.fake_y, self.y)
