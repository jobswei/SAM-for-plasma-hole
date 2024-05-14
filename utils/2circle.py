
import cv2
import os
import os.path as osp
import numpy as np

data_root="outputs/picts"
time=8
for space in range(570,800):
    img=cv2.imread(osp.join(data_root,f"{time}_{space}.jpg"))
    img=np.array(img)
    img.shape