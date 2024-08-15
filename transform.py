import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import ToTensor, Resize
from torchvision.transforms.functional import adjust_brightness, adjust_contrast

def resize(img: Tensor, width: int = 1024):
    return Resize((width, width))(img)

def rebright(img: Tensor, bright_factor: float = 1.5):
    return adjust_brightness(img, bright_factor)

def contrast(img: Tensor, contrast_factor = 1.5):
    return adjust_contrast(img, contrast_factor)

def read_tensor(img_path: str):
    return ToTensor()(cv2.imread(img_path))



# has to be run with an event loop 
def do_transform(img_path: str):
    img = read_tensor(img_path)
    img = resize(img)
    img = rebright(img)
    img = contrast(img)
    # other transforms

    # convert to vram?? turn into tensor/??
    return (img_path, img)





     