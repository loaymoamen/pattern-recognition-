import numpy as np
from PIL import Image

img = Image.open('8-2.jpg').convert('1')
arr = np.array(img)
flat_arr = arr.ravel()
