import os.path
import glob
from torch.utils.data import Dataset
from .dng_image import DNG
import numpy as np
import torch
import scipy.io
import cv2


# provide images with 00XXX.dng 

class DNGDataset(Dataset):
    """ 
        ARW Dataset. 
        Dataset with a pair of images
    """
    
    x_path = ''
    x_images = None

    # Define how many pair types we have 
    
    number_of_images = 0

    def __init__(self, manager, x_folder):
        super().__init__()

        self.manager = manager
        self.data_dir = manager.get_data_dir()

        # Sanity checks 
        if not isinstance(x_folder, str):
            raise Exception("x_path must be a string")
        if not os.path.isdir(self.data_dir+x_folder):
            raise Exception(f"x_path not found: {self.data_dir+x_folder}")

        self.x_path = f"{self.data_dir}{x_folder}/"
        
        x_data = glob.glob(f'{self.x_path}*.dng')
        self.x_ids = np.unique([int(os.path.basename(res)[0:5]) for res in x_data])

        max_size = manager.get_options().get("max_dataset_size")
        if max_size != 0 and not isinstance(max_size, int):
            raise Exception("Max size must be int")
        
        # If there is a max limit to the data size, we randomly pick an amount of them 

        if max_size:
            self.x_ids = np.random.choice(self.x_ids, max_size)            
            self.x_images = np.array([None] * max_size)
            
        else:
            self.x_images = np.array([None] * len(self.x_ids))

                
        manager.get_logger("system").info(f"Begin dng dataset | 0 images")

        self.load()    

        manager.get_logger("system").info(f"End ARW dataset | {self.number_of_images} images")

    def load(self):
        """
            Load images from the x and y paths.
            Images must be ARW files and have XXXXX_00_XX.ARW naming
        """
        self.number_of_pairs = 0
        count = 0
        for index, x_id in enumerate(self.x_ids):
            
            self.manager.get_logger("system").info(f"Loaded {self.number_of_images} images")
            
            x_file = glob.glob(self.x_path + '%05d.dng'%x_id)

            for x_path in x_file:
                
                dng = DNG(x_path)
                dng.pack(100)

                self.x_images[index] = dng

                self.number_of_images += 1


    def size(self):
        """Returns the number of images in the dataset"""
        return self.number_of_images

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        
        if self.manager.is_train:
            return self.get_image_patch(index)        
        else:
            return self.get_image(index)

    def get_image(self, index):
        
        x_image = self.x_images[index].get()
        
        H, W, D = x_image.shape

        xx = np.random.randint(0, W - 2048)
        yy = np.random.randint(0, H - 1024)
        
        x_patch = x_image[yy:yy + 1024, xx:xx + 2048, :]

        x_patch = np.minimum(x_patch, 1.0)
        x_patch = np.transpose(x_patch, (2, 0, 1))

        # Unpack before returning
        return x_patch

    def get_image_patch(self, index):
        """Get an image patch"""

        x_image = self.x_images[index].get()

    
        H, W, D = x_image.shape

        xx = np.random.randint(0, W - self.patch_size)
        yy = np.random.randint(0, H - self.patch_size)

        x_patch = x_image[yy:yy + self.patch_size, xx:xx + self.patch_size, :]
    

    def __len__(self):
        """We return the total number of counted pairs"""
        return self.number_of_images

