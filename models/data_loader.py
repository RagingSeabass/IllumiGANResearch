from torch.utils.data import Dataset
import torch
import glob, os, sys
import numpy as np
import rawpy


class LearningToSeeInTheDarkImagePair():
    """Save the mapping from short to long exposure"""
    
    ratio_key   = 0
    short_index = 0
    long_index  = 0

    def __init__(self, short_exposure, long_exposure, ratio_key):
        self.short_index    = short_exposure
        self.long_index     = long_exposure
        self.ratio_key      = ratio_key

    def __str__(self):
        return f"Long ID {self.long_id} - Short ID {self.short_id}/{self.ratio_key}"

class LearningToSeeInTheDarkDataset(Dataset):
### Learning to see in the dark dataset 
### Contains two folders 
### - short (x)
### - long  (y)
    def __init__(self, path_to_data, patch_size=512, debug=False):
        super(LearningToSeeInTheDarkDataset, self).__init__()

        self.patch_size = patch_size

        # Lets prepare to load all images into memory for fast training.
        if debug:
            data_short  = glob.glob(f'{path_to_data}short/000[0-4][0-9]*.ARW')
        else:
            data_short  = glob.glob(f'{path_to_data}short/0*.ARW')

        self.short_ids = np.unique([int(os.path.basename(res)[0:5]) for res in data_short])
        self.short_ids.sort()

        self.short_path = f"{path_to_data}short/"

        # Ground truth images
        self.long_path = f"{path_to_data}long/"
        
        # We want to load data into memory 
        self.long_images    = np.array([None] * 6000)
        
        self.short_image_pairs = np.array([None] * 6000)

        self.short_images = {}
        self.short_images['100'] = np.array([None] * len(self.short_ids))
        self.short_images['250'] = np.array([None] * len(self.short_ids))
        self.short_images['300'] = np.array([None] * len(self.short_ids))
        
        print("Loading data into memory", file=sys.stderr, flush=True)  
        self.short_count = 0

        for index, t_id in enumerate(self.short_ids):
            
            short_files = glob.glob(self.short_path + '%05d_00*.ARW'%t_id)
            long_files = glob.glob(self.long_path + '%05d_00*.ARW'%t_id)

            # Get the exposure number from photo title
            long_path = long_files[0]
            _, long_fn = os.path.split(long_path)
            long_exposure =  float(long_fn[9:-5])

            # Load raw file into memory
            long_raw = rawpy.imread(long_path)
            im = long_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            self.long_images[index] = np.expand_dims(np.float32(im/65535.0) , axis = 0)

            for short_path in short_files:
                # count for each type of image for total number of short exposure images
                
                _, short_fn = os.path.split(short_path)
                short_exposure =  float(short_fn[9:-5])

                ratio = min(long_exposure/short_exposure, 300)
                ratio_key = str(ratio)[0:3]

                self.short_image_pairs[self.short_count] = LearningToSeeInTheDarkImagePair(index, index, ratio_key)

                t_raw = rawpy.imread(short_path)
                self.short_images[ratio_key][index] = np.expand_dims(self.pack_raw(t_raw) , axis=0) * ratio

                self.short_count += 1

            if index % 10 == 0:
                print("Loaded %d ids "%index, file=sys.stderr, flush=True)

        print("Completed loading data into memory", file=sys.stderr, flush=True)

    def __getitem__(self, index):
        
        imagepair = self.short_image_pairs[index]

        rk = imagepair.ratio_key
        sindex = imagepair.short_index
        lindex = imagepair.long_index

        # crop
        H = self.short_images[rk][sindex].shape[1]
        W = self.short_images[rk][sindex].shape[2]

        xx = np.random.randint(0, W - self.patch_size)
        yy = np.random.randint(0, H - self.patch_size)

        short_patch = self.short_images[rk][sindex][:, yy:yy + self.patch_size, xx:xx + self.patch_size, :]
        long_patch  = self.long_images[lindex][:, yy * 2:yy * 2 + self.patch_size * 2, xx * 2:xx * 2 + self.patch_size * 2, :]

        if np.random.randint(2) == 1:  # random flip
            short_patch = np.flip(short_patch, axis=1)
            long_patch = np.flip(long_patch, axis=1)
        if np.random.randint(2) == 1:
            short_patch = np.flip(short_patch, axis=2)
            long_patch = np.flip(long_patch, axis=2)
        if np.random.randint(2) == 1:  # random transpose
            short_patch = np.transpose(short_patch, (0, 2, 1, 3))
            long_patch = np.transpose(long_patch, (0, 2, 1, 3))

        short_patch = np.minimum(short_patch,1.0)
        long_patch  = np.maximum(long_patch, 0.0)

        short_patch = short_patch[0]
        long_patch = long_patch[0]

        return short_patch, long_patch

    # Use rawpy to get pictures
    def pack_raw(self, raw):
        # Pack Bayer image to 4 channels
        im = raw.raw_image_visible.astype(np.float32)

        # Subtract the black level
        # 16383 == 2^14 (data is 14 bits)
        # 512 is hardware specific to the camera 
        im = np.maximum(im - 512, 0) / (16383 - 512)

        im = np.expand_dims(im, axis=2)
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]

        out = np.concatenate((im[0:H:2, 0:W:2, :],
                            im[0:H:2, 1:W:2, :],
                            im[1:H:2, 1:W:2, :],
                            im[1:H:2, 0:W:2, :]), axis=2)
        return out
            

    def __len__(self):
        return self.short_count




        

