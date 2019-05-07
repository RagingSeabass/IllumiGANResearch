from torch.optim import lr_scheduler
from torch.nn import init
import torch
import numpy as np
from PIL import Image


def get_lr_scheduler(optim, hyperparameters):
    """Returns the learning rate scheduler"""

    if hyperparameters.get("lr_policy") == 'linear':
        def lamda_(epoch):
            """Define lamda function for scheduler"""
            return 1.0 - max(0, epoch - hyperparameters.get("epoch_iterations")) / float(hyperparameters.get("epoch_decaying_iterations") + 1)     
        scheduler = lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lamda_)
    else:
        return NotImplementedError('lr_policy not implemented')

    return scheduler


def init_weights(net):
    """Initialize weights in a network"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, 0.02) # Torch init function
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1: 
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func) # apply initialization

def init_network(net, gpu_ids=[]):
    """Initialize network"""
    init_weights(net)
    return net


# DO NOT CALL A BATCH ONLY SINGLE IMAGE
def tensor2img(input_img):
    if not isinstance(input_img, np.ndarray):
        if isinstance(input_img, torch.Tensor):  # get the data from a variable
            image_tensor = input_img.data
        else:
            return input_img

        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array

        # FROM C,H,W to H,W,C
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_img
    return image_numpy.astype(np.uint8)