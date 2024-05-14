import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import matplotlib

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

    