
import sys
sys.path.append('../ZoeDepth/')

import cv2
import numpy as np

import torch
from zoedepth.utils.misc import get_image_from_url, colorize
from PIL import Image
import matplotlib.pyplot as plt


def getDepthFromImage(zoe,img) :
    img = Image.open(img)
    zoe = zoe.to("cuda")
    depth = zoe.infer_pil(img)
    # zoe = zoe.to("cpu") # Free up GPU memory after inference
    return depth

# img should read from PIL.Image
def getDepth(img,DEVICE='cuda'): # Takes Image , Points, DEVICE as input
    zoe = torch.hub.load("../ZoeDepth/", "ZoeD_NK", source="local", pretrained=True)
    zoe = zoe.to(DEVICE)
    img = Image.open(img)

    depth = zoe.infer_pil(img)

    zoe = zoe.to("cpu") # Free up GPU memory after inference

    del zoe
    return depth

def main():
    zoe = torch.hub.load("../ZoeDepth/", "ZoeD_NK", source="local", pretrained=True)
    zoe = zoe.to("cuda")
    FileName = 'frame_00001.jpg'
    img = Image.open('../EinstienVision/Data/Images/Images_1/'+FileName)

    depth = zoe.infer_pil(img)

    colored_depth = colorize(depth)
    cv2.imwrite("img.jpg",colored_depth)
    fig, axs = plt.subplots(1,2, figsize=(15,7))
    for ax, im, title in zip(axs, [img, colored_depth], ['Input', 'Predicted Depth']):
        ax.imshow(im)
        ax.axis('off')
        ax.set_title(title)

if __name__ == '__main__':
    main()