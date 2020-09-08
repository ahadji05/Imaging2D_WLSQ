import numpy as np
from PIL import Image

def readVelocityModel(file_vel):
    img = Image.open(file_vel)
    img = np.array(img.getdata()).reshape(img.size[1], img.size[0]) # note the reverse size dimensionality of image and matrix
    img = np.asarray(img, dtype=np.float32) # change datatype to float
    if __debug__:
        print(file_vel,"size :",img.shape)
    return img
