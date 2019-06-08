import os, time, scipy.io
from skimage.measure import compare_ssim as ssim
import numpy as np
import rawpy
import glob
import math
from PIL import Image

imgs = glob.glob('../../../expmobile/4500/test/*.png')

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssimm(img1, img2):
    return ssim(img1, img2, data_range=255, multichannel=True)

su = 0
ss = 0

for i, gt in enumerate(imgs):
    img_gt = Image.open(gt)
    img_out = Image.open(gt)

    img_gt = img_gt.crop((0,0,2048,1024))
    img_out = img_out.crop((2048,0,4096,1024))

    d=psnr(np.asarray(img_gt),np.asarray(img_out))
    s=ssimm(np.asarray(img_gt),np.asarray(img_out))

    ss = ss + s
    su = su + d

    print(i)
    
print(su / len(imgs))
print(ss / len(imgs))