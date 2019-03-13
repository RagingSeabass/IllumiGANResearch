import os.path
import glob
from data.base_dataset import BaseDataset
from data.arw_image import ARW
import numpy as np


class ARWDataset(BaseDataset):
    """ 
        ARW Dataset. 
        Dataset with a pair of images
    """
    
    x_path = ''
    y_path = ''

    x_ids = None
    x_images = {}
    y_images = None

    # Define how many pair types we have 
    xy_pairs = None
    pair_types = 3

    number_of_pairs = 0

    def __init__(self, opt, data_dir, x_folder, y_folder):
        BaseDataset.__init__(self, opt, data_dir)

        # Sanity checks 
        if not isinstance(x_folder, str):
            raise Exception("x_path must be a string")
        if not isinstance(y_folder, str):
            raise Exception("y_path must be a string")
        if not os.path.isdir(data_dir+x_folder):
            raise Exception(f"x_path not found: {data_dir+x_folder}")
        if not os.path.isdir(data_dir+y_folder):
            raise Exception(f"x_path not found: {data_dir+y_folder}")

        self.x_path = f"{data_dir}{x_folder}/"
        self.y_path = f"{data_dir}{y_folder}/"

        x_data = glob.glob(f'{self.x_path}0*.ARW')
        self.x_ids = np.unique([int(os.path.basename(res)[0:5]) for res in x_data])

        max_size = opt.get("max_dataset_size")
        if max_size != None and not isinstance(max_size, int):
            raise Exception("Max size must be int")

        if max_size:
            self.x_ids = np.random.choice(self.x_ids, max_size)
            self.y_images = np.array([None] * max_size)
            self.x_images['100'] = np.array([None] * max_size)
            self.x_images['250'] = np.array([None] * max_size)
            self.x_images['300'] = np.array([None] * max_size) 
            self.xy_pairs = np.array([None] * (max_size*self.pair_types))
        else:
            self.y_images = np.array([None] * len(self.x_ids)) 
            self.x_images['100'] = np.array([None] * len(self.x_ids))
            self.x_images['250'] = np.array([None] * len(self.x_ids))
            self.x_images['300'] = np.array([None] * len(self.x_ids)) 
            self.xy_pairs = np.array([None] * (len(self.x_ids)*self.pair_types))
        
        self.load()    


    def load(self):
        """
            Load images from the x and y paths.
            Images must be ARW files and have XXXXX_00_XX.ARW naming
        """
        self.number_of_pairs = 0
        for index, x_id in enumerate(self.x_ids):
            x_files = glob.glob(self.x_path + '%05d_00*.ARW'%x_id)
            y_files = glob.glob(self.y_path + '%05d_00*.ARW'%x_id)

            y_exposure = self.get_exposure(y_files[0])   

            # post process out image 
            arw = ARW(y_files[0])
            arw.postprocess()
            self.y_images[index] = arw

            for x_path in x_files:
                x_exposure = self.get_exposure(x_path)

                ratio = min(y_exposure/x_exposure, 300)
                ratio_key = str(ratio)[0:3]

                self.xy_pairs[self.number_of_pairs] = ExposureImagePair(index, ratio_key)

                # Pack image into 4 channels
                arw = ARW(x_path)
                arw.pack(ratio)

                self.x_images[ratio_key][index] = arw
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
        return self.get_image_patch(pair.index, pair.ratio_key)        

    def get_image_patch(self, index, ratio_key):
        """Get an image patch"""
        # crop
        x_image = self.x_images[ratio_key][index].get()
        y_image = self.y_images[index].get()
        
        _, H, W, D = x_image.shape

        xx = np.random.randint(0, W - self.patch_size)
        yy = np.random.randint(0, H - self.patch_size)

        x_patch = x_image[:, yy:yy + self.patch_size, xx:xx + self.patch_size, :]
        y_patch  = y_image[:, yy * 2:yy * 2 + self.patch_size * 2, xx * 2:xx * 2 + self.patch_size * 2, :]

        # Data augmentations
        if np.random.randint(2) == 1:  # random flip
            x_patch = np.flip(x_patch, axis=1)
            y_patch = np.flip(y_patch, axis=1)
        if np.random.randint(2) == 1:
            x_patch = np.flip(x_patch, axis=2)
            y_patch = np.flip(y_patch, axis=2)
        if np.random.randint(2) == 1:  # random transpose
            x_patch = np.transpose(x_patch, (0, 2, 1, 3))
            y_patch = np.transpose(y_patch, (0, 2, 1, 3))
        
        x_patch = np.minimum(x_patch,1.0)
        y_patch  = np.maximum(y_patch, 0.0)

        # Unpack before returning
        return x_patch[0], y_patch[0]

    def __len__(self):
        """We return the total number of counted pairs"""
        return self.number_of_pairs


class ExposureImagePair():
    """Save the mapping from short to long exposure"""
    
    ratio_key   = 0
    index = 0

    def __init__(self, index, ratio_key):
        self.index      = index
        self.ratio_key  = ratio_key

    def __str__(self):
        return f"Long ID {self.long_id} - Short ID {self.short_id}/{self.ratio_key}"
    







        
        