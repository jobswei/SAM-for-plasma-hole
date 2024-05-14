# %%
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
import pickle
import os.path as osp
import tqdm

# %%
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
def getImage(image,resize=-1):
    cmap = matplotlib.colormaps["viridis"]
    norm = plt.Normalize(image.min(), image.max())
    rgba_image = cmap(norm(image))
    rgb_image = np.array(rgba_image[:, :, :3])
    rgb_array_scaled = (rgb_image * 255).astype(np.uint8)
    if resize==-1:
        return rgb_array_scaled
    image_pil = Image.fromarray(rgb_array_scaled)
    new_size = resize
    resized_image_pil = image_pil.resize(new_size, Image.LANCZOS)
    return np.array(resized_image_pil)
def show(image):
    plt.figure()
    plt.axis("off")
    plt.imshow(image)
    plt.show()

def to_origin(mask,size=(401,31)):
    mask_img=Image.fromarray(mask)
    mask_img=mask_img.resize(size,Image.LANCZOS)
    mask_img=np.array(mask_img)
    return mask_img

def display_result(image, mask):
    plt.figure()
    plt.imshow(image)
    # show_box(input_box, plt.gca())
    show_mask(mask, plt.gca())
    # show_points(input_point, input_label, plt.gca())
    # plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    # plt.savefig(f"./outputs/res{i}.jpg")
    plt.show()

    

# %%
def S_inte(f, mask:np.ndarray, x_range=[0,2*np.pi],y_range=[-0.1,0.1]):
    y_len, x_len = mask.shape
    x_lis=np.linspace(x_range[0],x_range[1],x_len)
    y_lis=np.linspace(y_range[0],x_range[0],y_len)
    dx=(x_range[1]-x_range[0])/x_len
    dy=(y_range[1]-y_range[0])/y_len

    f_res=f(x_lis,y_lis)
    summ=0
    for y_ind in range(y_len):
        for x_ind in range(x_len):
            if mask[y_ind][x_ind]:
                summ+=f_res[y_ind][x_ind]*dx*dy
    return summ

def FL(f_trap,mask_origin, mgL):
    def numerator(x,y):
        return mgL*np.sin(x)*f_trap
    def denominator(x,y):
        return f_trap
    return S_inte(numerator,mask_origin)/S_inte(denominator,mask_origin)

# %%
sam = sam_model_registry["vit_h"](checkpoint="/home/wzy/segment-anything/checkpoints/sam_vit_h_4b8939.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)

mask_generator = SamAutomaticMaskGenerator(sam)

# %%
def get_mask(f_trap, predictor, show=False, save=False, save_dir=None):
    image=getImage(f_trap,(256,256))
    # plt.axis("off")
    # plt.imshow(image)
    # image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)
    input_box = np.array([64,64,192,192])
    input_point = np.array([[128, 128]])
    input_label = np.array([1]) # 前景与背景
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        # box=input_box[None, :],
        multimask_output=False,
    )
    # print(masks.shape) # (number_of_masks) x H x W
    mask=masks[0]
    score=scores[0]
    is_construct=True

    # print(np.sum(mask[:,-3:-1]) + np.sum(mask[:,0:2])+np.sum(mask[-3:-1,:])+np.sum(mask[0:2,:]))
    # print((mask.shape[0]+mask.shape[1])/8)
    if np.sum(mask[:,-3:-1]) + np.sum(mask[:,0:2])+np.sum(mask[-3:-1,:])+np.sum(mask[0:2,:])\
        >(mask.shape[0]+mask.shape[1])/8 or score<0.9:    
        is_construct=False
        # print("false")
    if show:
        plt.figure()
        plt.imshow(image)
        # show_box(input_box, plt.gca())
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        # plt.savefig(f"./outputs/res{i}.jpg")
        plt.show()
    if save:
        plt.figure()
        plt.imshow(image)
        # show_box(input_box, plt.gca())
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(save_dir)
        # plt.show()
    return to_origin(mask,size=f_trap.shape[::-1]), is_construct

def get_FL(f_trap, mask_origin, mgL):
    f_trap=np.transpose(f_trap)
    mask_origin=np.transpose(mask_origin)
    fl=FL(f_trap,mask_origin, mgL)
    # print(fl)
    return fl


# %%
dist=np.load("/home/wzy/segment-anything/data/newd.npy")
print(dist.shape)

# %%
wb_lis=np.load("/home/wzy/segment-anything/data/wb2.npy")
print(wb_lis.shape)

# %%
res_dir="/home/wzy/segment-anything/outputs/mask_res"
for time in range(dist.shape[0]):
    print(f"timestamp:{time}")
    for space in tqdm.tqdm(range(dist.shape[1])):
        f_trap=dist[time,space,:,:]
        mask, is_construct= get_mask(f_trap,predictor)
        with open(osp.join("/home/wzy/segment-anything/outputs/mask_res",f"{time}_{space}_{is_construct}.pkl"),"wb") as fp:
            pickle.dump((mask,is_construct),fp)
# if is_construct:
#     fl=get_FL(f_trap, mask, mgL=1)
#     print(fl)



