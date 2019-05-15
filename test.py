

from PIL import Image
from PIL import ImageEnhance
import numpy as np

png = Image.open('test.png')

print(png.format, png.size, png.mode)
test = np.asarray(png)


enhancer = ImageEnhance.Brightness(png)
#enhancer.enhance(100).show("Sharpness 0.8")