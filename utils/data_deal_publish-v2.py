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
import os
import os.path as osp
import tqdm
from scipy.ndimage import label


# %%
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
def getImage(image,resize=(256,256)):
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
def find_connected_masks(mask_matrix, target_point):
    # 标记连通区域
    labeled_matrix, num_labels = label(mask_matrix)

    # 找到与给定点相连的mask
    connected_mask = (labeled_matrix == labeled_matrix[target_point])
    # print(type(connected_mask))
    # print(connected_mask)

    return connected_mask

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

def FL(f_trap,mask_origin, mgL,x_range=[0,2*np.pi]):
    def numerator(x,y):
        return mgL*np.sin(x)*f_trap
    def denominator(x,y):
        return f_trap
    return S_inte(numerator,mask_origin, x_range=x_range)/S_inte(denominator,mask_origin,x_range=x_range)

# %%
sam = sam_model_registry["vit_h"](checkpoint="/home/wzy/segment-anything/checkpoints/sam_vit_h_4b8939.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)

mask_generator = SamAutomaticMaskGenerator(sam)

# %%
def find_min(arr,num_candidate=10):
    flattened_indices = np.argsort(arr.flatten())

    # 取前10个索引
    top_10_indices = flattened_indices[:num_candidate]

    # 将一维索引转换为二维索引
    row_indices, col_indices = np.unravel_index(top_10_indices, arr.shape)

    # 输出前10小的数和它们的位置
    # print("Top 10 smallest values:", arr[row_indices, col_indices])
    # print("Corresponding indices:", list(zip(row_indices, col_indices)))
    return arr[row_indices, col_indices], list(zip(row_indices, col_indices))
def distance(point1,point2):
    delta=np.array(point1)-np.array(point2)
    return delta[0]**2+delta[1]**2
def get_fix_point(f_trap,num_candidate=10):
    center=np.array(f_trap.shape)/2
    _, positions=find_min(f_trap,num_candidate)
    dists=np.array(list(map(lambda x:distance(center,x), positions)))
    min_index = np.argmin(dists)
    return list(positions[min_index])
def get_fix_points(f_trap,k=4,num_candidate=10):
    center=np.array(f_trap.shape)/2
    _, positions=find_min(f_trap,num_candidate)
    dists=np.array(list(map(lambda x:distance(center,x), positions)))
    dist_sort=np.argsort(dists)
    top_k=dist_sort[:k]
    print(top_k)
    return [list(positions[i]) for i in top_k]

def multi_point(input_point, radius=5, num_points=5):
    theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    x = input_point[0] + radius * np.cos(theta)
    y = input_point[1] + radius * np.sin(theta)
    points = list(np.column_stack((x, y)))
    # points.append(input_point)
    return points


# %%
def _get_mask(image, predictor,thre=0.8,div=8,input_point=[[145,128]], show=False, save=False, save_dir=None):
    # image=getImage(f_trap,(256,256))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)
    input_point = np.array(input_point)
    input_label = np.array([1]*len(input_point)) # 前景与背景
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    # print(masks.shape) # (number_of_masks) x H x W
    mask=masks[0]
    score=scores[0]
    is_construct=True
    center_point=tuple(np.mean(input_point,axis=0).astype(int))
    mask=find_connected_masks(mask,center_point[::-1]) # 注意 网络输入point和数组point的xy索引是不一样的

    # print(np.sum(mask[:,-3:-1]) + np.sum(mask[:,0:2])+np.sum(mask[-3:-1,:])+np.sum(mask[0:2,:]))
    # print((mask.shape[0]+mask.shape[1])/8)
    if np.sum(mask[:,-3:-1]) + np.sum(mask[:,0:2])+np.sum(mask[-3:-1,:])+np.sum(mask[0:2,:])\
        >(mask.shape[0]+mask.shape[1])/div or score<thre:    
        is_construct=False
        # print("false")
    if show:
        plt.figure()
        plt.imshow(image)
        # show_box(input_box, plt.gca())
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Score: {score:.3f}", fontsize=10)
        plt.axis('off')
        # plt.savefig(f"./outputs/res{i}.jpg")
        plt.show()
    if save:
        plt.figure()
        plt.imshow(image)
        # show_box(input_box, plt.gca())
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Score: {score:.3f} {is_construct}", fontsize=10)
        plt.axis('off')
        plt.savefig(save_dir,bbox_inches='tight')
        plt.close()
        # plt.show()
    return mask, is_construct

def get_mask(delta_f, predictor,multi_points=True,fix_cneter_x=True,radias=25,num_points=3,num_candidate=10,thre=0.6,div=10000, show=False, save=False, save_dir=None):

    delta_f_trap_2cir=np.concatenate((delta_f,delta_f),axis=0)
    delta_f_trap_2cir_resize=to_origin(delta_f_trap_2cir,(256,256*2))
    image=getImage(delta_f_trap_2cir_resize,(256,256*2))
    # plt.imshow(image)

    center_point=get_fix_point(delta_f_trap_2cir_resize,num_candidate=num_candidate)
    center_point.reverse()
    if fix_cneter_x:
        center_point[0]=delta_f_trap_2cir_resize.shape[1]//2
    if multi_points:
        input_point=multi_point(center_point,radius=radias,num_points=num_points)
        temp=[]
        for point in input_point:
            if point[1]<0 or point[1]>delta_f_trap_2cir_resize.shape[0]-1 or \
               point[0]<0 or point[0]>delta_f_trap_2cir_resize.shape[1]-1 or \
                delta_f_trap_2cir_resize[int(point[1]),int(point[0])]>-0.0099:#0.1*delta_f_trap_2cir_resize[int(center_point[1]),int(center_point[0])]:
                continue
            temp.append(point)
        input_point=temp
    else:
        input_point=[center_point]
    if not input_point:
        input_point=[center_point]
    # if 
    # print(input_point)
    # plt.imshow(image)
    mask, is_construct=_get_mask(image,predictor,thre=thre,div=div,input_point=input_point,show=show,save=save,save_dir=save_dir)
    return to_origin(mask,delta_f_trap_2cir.shape[::-1]), is_construct # 图片xy的顺序和数组xy的顺序是反的
def get_mask2(delta_f, predictor,multi_points=True,fix_cneter_x=True,radias=20,num_points=3,num_candidate=10,thre=0.8,div=8, show=False, save=False, save_dir=None):

    delta_f_trap_2cir=np.concatenate((delta_f,delta_f),axis=0)
    delta_f_trap_2cir_resize=to_origin(delta_f_trap_2cir,(256,256*2))
    image=getImage(delta_f_trap_2cir_resize,(256,256*2))
    # plt.imshow(image)

    center_point=get_fix_points(delta_f_trap_2cir_resize,k=10,num_candidate=num_candidate)
    for i in center_point:
        i.reverse()
    # if fix_cneter_x:
    #     center_point[0]=delta_f_trap_2cir_resize.shape[1]//2
    # if multi_points:
    #     input_point=multi_point(center_point,radius=radias,num_points=num_points)
    # else:
    input_point=center_point
    # print(input_point)
    # plt.imshow(image)
    mask, is_construct=_get_mask(image,predictor,thre=thre,div=div,input_point=input_point,show=show,save=save,save_dir=save_dir)
    return to_origin(mask,delta_f_trap_2cir.shape[::-1]), is_construct # 图片xy的顺序和数组xy的顺序是反的
def get_FL(f_trap, mask_origin, mgL,x_range=[0,2*np.pi]):
    f_trap=np.transpose(f_trap)
    mask_origin=np.transpose(mask_origin)
    fl=FL(f_trap,mask_origin, mgL,x_range=x_range)
    # print(fl)
    return fl
def get_alpha(f_trap, mask_origin, mgL,x_range=[0,2*np.pi]):
    f_trap=np.transpose(f_trap)
    mask_origin=np.transpose(mask_origin)
    assert f_trap.shape==mask_origin.shape, f"{f_trap.shape}!={mask_origin.shape}"
    fl=FL(f_trap,mask_origin, mgL,x_range=x_range)
    # print(fl)
    return fl/mgL


# %%
dist=np.load("/home/wzy/segment-anything/data/newd.npy")
print(dist.shape)
feq_lis=np.load("/home/wzy/segment-anything/data/feq.npy")
feq_lis=np.transpose(feq_lis)
print(feq_lis.shape)
wb_lis=np.load("/home/wzy/segment-anything/data/wb2.npy")
print(wb_lis.shape)
k_lis=np.load("/home/wzy/segment-anything/data/kmode.npy")
print(k_lis.shape)

# %%


# %%

# %%
res_dir="/home/wzy/segment-anything/outputs/mask_res_2"
os.makedirs(res_dir, exist_ok=True)
for time in [7]:
    print(f"timestamp:{time}")
    for space in tqdm.tqdm(range(400,1000)):
        f_trap=dist[time,space,:,:]
        delta_f=f_trap-feq_lis[space]
        mask,is_construct=get_mask(delta_f,predictor,num_points=4,radias=25,div=20,multi_points=True,fix_cneter_x=False,save=True,save_dir=osp.join("/home/wzy/segment-anything/outputs/",f"picts_2/{time}_{space}_2.jpg"))
        # print(is_construct)
        with open(osp.join(res_dir,f"{time}_{space}_{is_construct}.pkl"),"wb") as fp:
            pickle.dump((mask,is_construct),fp)
# if is_construct:
#     fl=get_FL(f_trap, mask, mgL=1)
#     print(fl)




# %%
