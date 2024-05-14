# %%
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import matplotlib.pyplot as plt
from demo_utils import *
import tqdm

# %%
sam = sam_model_registry["vit_h"](checkpoint="/home/wzy/segment-anything/checkpoints/sam_vit_h_4b8939.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)

# %%
def inference(image,out_path):
    predictor.set_image(image)
    input_point = np.array([[128, 128]])
    input_label = np.array([1]) # 前景与背景
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    # print(masks.shape) # (number_of_masks) x H x W

    for i, (mask, score) in enumerate(zip(masks, scores)):
        # print(np.sum(mask[:,-3:-1]) + np.sum(mask[:,0:2])+np.sum(mask[-3:-1,:])+np.sum(mask[0:2,:]))
        # print((mask.shape[0]+mask.shape[1])/8)

        plt.figure()
        plt.imshow(image)
        if not (np.sum(mask[:,-3:-1]) + np.sum(mask[:,0:2])+np.sum(mask[-3:-1,:])+np.sum(mask[0:2,:])\
            >(mask.shape[0]+mask.shape[1])/8 or score<0.9):
            # print("false")
            show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(out_path)


# %%


# %%
data_root="/home/wzy/segment-anything/"
for i in tqdm.tqdm(range(500,600)):
    image = cv2.imread(data_root+f'data/images/{i}.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inference(image,data_root+f"./outputs/res_{i}.jpg")


