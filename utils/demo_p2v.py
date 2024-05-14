from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

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

sam = sam_model_registry["vit_h"](checkpoint="/home/wzy/segment-anything/checkpoints/sam_vit_h_4b8939.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)

dist=np.load("/home/wzy/segment-anything/data/newd.npy")



output_file=""
#########################
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编解码器
width, height = 256,256
out = cv2.VideoWriter(output_file, fourcc, 30, (width, height))  # 设置帧率为 30
frame_count = 0
frame_sampling_rate =1
#########################


for time in tqdm.tqdm(range()):

    image=dist[time,600,:,:]
    image=getImage(image,(256,256))

    # image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)
    input_point = np.array([[128, 128]])
    input_label = np.array([1]) # 前景与背景
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    print(masks.shape) # (number_of_masks) x H x W

    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask=show_mask(mask)
        res=image+mask
        out.write(res)
out.release()

        