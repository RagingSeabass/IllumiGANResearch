import rawpy
import numpy as np
from PIL import Image
import scipy.io
import cv2

class DNG():
    """"""

    path = ''
    data = None
    
    def __init__(self, path):
        self.path = path
        self.black_level = 535 #https://github.com/cchen156/Learning-to-See-in-the-Dark/issues/39
 
    def post(self):

        raw = rawpy.imread(self.path)
        self.data = raw.postprocess(half_size=True, no_auto_bright=True)
        

    def pack(self, ratio):
        
        raw = rawpy.imread(self.path)
        
        raw_matix = raw.raw_image_visible.astype(np.float32)
        

        # Subtract the black level
        # 16383 == 2^14 (Raw is 14 bits)
        # 512 is hardware specific to the camera
        # This normalizes the images!

        raw_matix = np.maximum(raw_matix - self.black_level, 0) / (16383 - self.black_level)
        raw_matix = np.expand_dims(raw_matix, axis=2)
        #raw_matix = np.rot90(raw_matix)

        print(raw_matix.shape)
        #scale_percent = 1.1 # percent of original size
        #width = int(raw_matix.shape[1] + 208)
        #height = int(raw_matix.shape[0] + 208)
        #dim = (width, height)

        #raw_matix = cv2.resize(raw_matix, dim)
        #raw_matix = np.expand_dims(raw_matix, axis=2)
        
        #raw_matix = cv2.resize(raw_matix, dsize=(3024, 4240))

        img_shape = raw_matix.shape
        H = img_shape[0]
        W = img_shape[1]
        D = img_shape[2]

        # We will now use the knowledge of bayers sensors 
        # We split by the colors into seperate matrixes 
        # We know that there are twice as many green, thus
        # we have 2 green matixes

        # Take every other 
        # [R,0,R,0,R]
        # [0,0,0,0,0]
        # [R,0,R,0,R]
        # [0,0,0,0,0]
        # [R,0,R,0,R]

        red = raw_matix[0:H:2, 0:W:2, :]

        # Take every other green 
        # [0,G,0,G,0]
        # [0,0,0,0,0]
        # [0,G,0,G,0]
        # [0,0,0,0,0]
        # [0,G,0,G,0]
        green1 = raw_matix[0:H:2, 1:W:2, :]

        # Take every other green 
        # [0,0,0,0,0]
        # [G,0,G,0,G]
        # [0,0,0,0,0]
        # [G,0,G,0,G]
        # [0,0,0,0,0]
        green2 = raw_matix[1:H:2, 0:W:2, :]

        # Take every other blue 
        # [0,0,0,0,0]
        # [0,B,0,B,0]
        # [0,0,0,0,0]
        # [0,B,0,B,0]
        # [0,0,0,0,0]
        blue = raw_matix[1:H:2, 1:W:2, :]

        # Shape: (1424, 2128, 4)
        image_matrix = np.concatenate((red, green1, blue, green2), axis=2)
        
        #self.data = np.expand_dims(image_matrix, axis=0) * ratio
        # Lets not expand the dimention as we want batch handling done by pytorch
        self.data = image_matrix * ratio
        
    def get(self):
        return self.data