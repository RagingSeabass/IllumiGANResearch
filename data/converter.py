import rawpy
import numpy as np
import glob
import imageio



x1_files = glob.glob('data/train/long/*.ARW')
x2_files = glob.glob('data/train/short/*.ARW')
x3_files = glob.glob('data/test/long/*.ARW')
x4_files = glob.glob('data/train/short/*.ARW')

for f in x1_files:
    with rawpy.imread(f) as raw:
        rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        imageio.imsave(f[:-4]+'.jpg', rgb)

for f in x2_files:
    with rawpy.imread(f) as raw:
        rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        imageio.imsave(f[:-4]+'.jpg', rgb)

for f in x3_files:
    with rawpy.imread(f) as raw:
        rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        imageio.imsave(f[:-4]+'.jpg', rgb)

for f in x4_files:
    with rawpy.imread(f) as raw:
        rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        imageio.imsave(f[:-4]+'.jpg', rgb)

