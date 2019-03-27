import rawpy
import numpy as np



class ARW():
    """"""

    path = ''
    data = None
    
    
    def __init__(self, path):
        self.path = path
        self.black_level = 512

    def postprocess(self):
        raw = rawpy.imread(self.path)
        # Spits out a 16 bit image! 
        img = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # Normalize image (by dividing with 2^16 = 65535) and wrap in list
        #self.data = np.expand_dims( np.float32( img / 65535.0 ) , axis = 0)
        self.data = np.float32( img / 65535.0 )

    def pack(self, ratio):
        raw = rawpy.imread(self.path)
        raw_matix = raw.raw_image_visible.astype(np.float32)

        # Subtract the black level
        # 16383 == 2^14 (Raw is 14 bits)
        # 512 is hardware specific to the camera
        raw_matix = np.maximum(raw_matix - self.black_level, 0) / (16383 - self.black_level)
        raw_matix = np.expand_dims(raw_matix, axis=2)

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
