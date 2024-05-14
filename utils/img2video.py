
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from demo_utils import *
import tqdm
import os
import os.path as osp
data_root="/home/wzy/segment-anything/"
image_folder=osp.join(data_root,"outputs/picts_3")

image_0=osp.join(image_folder,"6_172_3.jpg")
frame = cv2.imread(image_0)
height, width, layers = frame.shape

# 定义视频编解码器并创建VideoWriter对象
# video_name = 'outputs/videos/thre0.8.avi'
# out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))

# for i in tqdm.tqdm(range(500,700)):
#     image = cv2.imread(data_root+f"outputs/res_{i}.jpg")
#     out.write(image)
# out.release()

video_name = 'outputs/videos/new.avi'
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))
time=6
for i in tqdm.tqdm(range(201)):
    image = cv2.imread(osp.join(image_folder,f"{time}_{i}_3.jpg"))
    out.write(image)
out.release()