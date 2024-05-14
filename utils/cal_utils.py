import numpy as np
from demo_utils import *

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

def get_mask(f_trap, predictor, show=True, save=False, save_dir=None):
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

    print(np.sum(mask[:,-3:-1]) + np.sum(mask[:,0:2])+np.sum(mask[-3:-1,:])+np.sum(mask[0:2,:]))
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
