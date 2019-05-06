import os.path
import glob
from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import crop
import scipy.io
from PIL import Image


class JPGDataset(Dataset):
    """ 
        ARW Dataset. 
        Dataset with a pair of images
    """
    
    x_path = ''
    y_path = ''

    x_ids = None
    x_images = {}
    x_images_processed = {}
    y_images = None

    # Define how many pair types we have 
    xy_pairs = None
    pair_types = 3

    number_of_pairs = 0

    transform_image = False

    def __init__(self, manager, x_folder, y_folder, transforms=False):
        super().__init__()

        self.manager = manager
        self.data_dir = manager.get_data_dir()
        self.transform_image = transforms;
        self.patch_size = manager.get_hyperparams().get("patch_size")

        # Sanity checks 
        if not isinstance(x_folder, str):
            raise Exception("x_path must be a string")
        if not isinstance(y_folder, str):
            raise Exception("y_path must be a string")
        if not os.path.isdir(self.data_dir+x_folder):
            raise Exception(f"x_path not found: {self.data_dir+x_folder}")
        if not os.path.isdir(self.data_dir+y_folder):
            raise Exception(f"x_path not found: {self.data_dir+y_folder}")

        self.x_path = f"{self.data_dir}{x_folder}/"
        self.y_path = f"{self.data_dir}{y_folder}/"

        x_data = glob.glob(f'{self.x_path}*.jpg')
        self.x_ids = np.unique([int(os.path.basename(res)[0:5]) for res in x_data])

        y_data = glob.glob(f'{self.y_path}*.jpg')
        self.y_ids = np.unique([int(os.path.basename(res)[0:5]) for res in y_data])

        max_size = manager.get_options().get("max_dataset_size")
        if max_size != 0 and not isinstance(max_size, int):
            raise Exception("Max size must be int")
        
        # If there is a max limit to the data size
        if max_size:
            self.x_ids = np.random.choice(self.x_ids, max_size)
            self.y_images = np.array([None] * max_size)
            self.x_images['100'] = np.array([None] * max_size)
            self.x_images['250'] = np.array([None] * max_size)
            self.x_images['300'] = np.array([None] * max_size)
            self.x_images_processed['100'] = np.array([None] * max_size)
            self.x_images_processed['250'] = np.array([None] * max_size)
            self.x_images_processed['300'] = np.array([None] * max_size) 
            self.xy_pairs = np.array([None] * (max_size * self.pair_types))
        else:
            self.y_images = np.array([None] * len(self.x_ids)) 
            self.x_images['100'] = np.array([None] * len(self.x_ids))
            self.x_images['250'] = np.array([None] * len(self.x_ids))
            self.x_images['300'] = np.array([None] * len(self.x_ids))
            self.x_images_processed['100'] = np.array([None] * len(self.x_ids))
            self.x_images_processed['250'] = np.array([None] * len(self.x_ids))
            self.x_images_processed['300'] = np.array([None] * len(self.x_ids)) 
            self.xy_pairs = np.array([None] * (len(self.x_ids)*self.pair_types))
        
        self.load()    

        # Final check
        if len(self.y_ids) < manager.get_hyperparams().get('batch_size'):
            raise Exception('Batch size must not be larger than number of data points!')

        manager.get_logger("system").info(f"Loaded ARW dataset | {self.number_of_pairs} image pairs")

    def load(self):
        """
            Load images from the x and y paths.
            Images must be JPG files and have XXXXX_00_XX.jpg naming
        """
        self.number_of_pairs = 0
        for index, x_id in enumerate(self.x_ids):
            
            x_files = glob.glob(self.x_path + '%05d_00*.jpg'%x_id)
            y_files = glob.glob(self.y_path + '%05d_00*.jpg'%x_id)

            y_exposure = self.get_exposure(y_files[0])   

            # post process out image 
            jpg = Image.open(y_files[0])
            self.y_images[index] = np.asarray(jpg)

            for x_path in x_files:
                
                x_exposure = self.get_exposure(x_path)

                ratio = min(y_exposure/x_exposure, 300)
                ratio_key = str(ratio)[0:3]

                self.xy_pairs[self.number_of_pairs] = ExposureImagePair(x_path, index, ratio_key)

                # Pack image into 4 channels
                
                # LETS HALF THE SIZES OF THE INPUT
                jpg = Image.open(x_path)
                W, H = jpg.size
                self.x_images[ratio_key][index] = np.asarray(jpg.resize((round(W/3),round(H/3)), Image.ANTIALIAS))
                self.x_images_processed[ratio_key][index] = np.asarray(jpg)

                self.number_of_pairs += 1

    def get_exposure(self, path):
        """Get exposure from title"""
        _, fn = os.path.split(path)
        return float(fn[9:-5])

    def pairs_size(self):
        """Returns the number of image pairs in the dataset"""
        return self.number_of_pairs

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        
        pair = self.xy_pairs[index]
        # returns X, Y

        if self.manager.is_train:
            
            x_image = self.x_images[pair.ratio_key][pair.index]
            x_image_processed = self.x_images_processed[pair.ratio_key][pair.index]
            y_image = self.y_images[pair.index]
            
            H, W, D = x_image.shape
            
            xx = np.random.randint(0, W - self.patch_size)
            yy = np.random.randint(0, H - self.patch_size)

            xxps = xx + self.patch_size
            yyps = yy + self.patch_size

            mult = 3
            
            xx2x = xx * mult
            xxps2x = xx2x + self.patch_size * mult
            yy2x = yy * mult
            yyps2x = yy2x + self.patch_size * mult

            x_patch = x_image[yy : yyps, xx : xxps, :]
            x_patch_processed = x_image_processed[yy2x : yyps2x, xx2x : xxps2x, :]     
            y_patch  = y_image[yy2x : yyps2x, xx2x : xxps2x, :]

            x_image = Image.fromarray(x_patch)
            x_image_processed = Image.fromarray(x_patch_processed)
            y_image = Image.fromarray(y_patch)

            transform_list = []
            if self.transform_image:

                if np.random.randint(2) == 1:
                    transform_list.append(transforms.RandomHorizontalFlip(1))
                    
                if np.random.randint(2) == 1:
                    transform_list.append(transforms.RandomVerticalFlip(1))
                    


                transform_list.append(transforms.ToTensor())
                
                transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
                
            tt = transforms.Compose(transform_list)
            
            return tt(x_image), tt(x_image_processed), tt(y_image)
       
        else:
            
            x_image = self.x_images[pair.ratio_key][pair.index]
            y_image = self.y_images[pair.index]
            
            transform_list = []
            transform_list.append(transforms.ToTensor())

            tt = transforms.Compose(transform_list)

            return tt(x_image).contiguous(), tt(y_image).contiguous()

    def __len__(self):
        """We return the total number of counted pairs"""
        return self.number_of_pairs

class ExposureImagePair():
    """Save the mapping from short to long exposure"""
    
    ratio_key   = 0
    index = 0
    x_path = 0

    def __init__(self, path, index, ratio_key):
        self.index      = index
        self.ratio_key  = ratio_key
        self.x_path = path

    def __str__(self):
        return f"Long ID {self.index} - Short ID {self.index}/{self.ratio_key}"
    







        
        