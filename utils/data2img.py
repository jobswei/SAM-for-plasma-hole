import cv2
import numpy as np
from demo_utils import *
import tqdm

dist=np.load("/home/wzy/segment-anything/data/newd.npy")
print(dist.shape)

for i in tqdm.tqdm(range(700,750)):
    image=dist[8,i,:,:]
    image=getImage(image,(256,256))
    plt.axis("off")
    plt.imshow(image)
    # image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"data/images/{i}.jpg",image)