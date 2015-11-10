import numpy as np
from PIL import Image

img = Image.open('ass1/data/8-2.jpg').convert('L')
img = img.resize((8,8), Image.ANTIALIAS)
arr = np.array(img)
flat_arr = arr.ravel()
flat_arr = 15 - (flat_arr/16)