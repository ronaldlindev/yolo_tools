import cv2
import numpy as np
import torch
from pathlib import Path
from torch import Tensor
from torchvision.transforms import ToTensor, Resize
import imutils
from torchvision.transforms.functional import adjust_brightness, adjust_contrast
from torchvision.utils import save_image
from skimage import exposure

def resize(img: Tensor, width: int = 1024):
    return Resize((width, width))(img)

def safe_resize(img: np.ndarray, width = 1024) -> Tensor:
    return ToTensor()(imutils.resize(img.permute(1,2,0).numpy(), width = width))
def rebright(img: Tensor, bright_factor: float = 1.5):
    return adjust_brightness(img, bright_factor)

def balance_gamma(img: Tensor, gamma: float = 0.5):
    return ToTensor()(exposure.adjust_gamma(img.permute(1,2,0).numpy(), gamma = gamma))

def contrast(img: Tensor, contrast_factor = 2.0):
    return adjust_contrast(img, contrast_factor)

def read_tensor(img_path: str):
    print('reading', img_path)
    return ToTensor()(cv2.imread(img_path))

def save_tensor(img : Tensor, img_path):
    print('writing', img_path)
    save_image(img, img_path)



# has to be run with an event loop 
def do_transform(img: Tensor, transforms: list, img_path: Path = None):
    if img_path:
        img = img_path
    for transform in transforms:
        img = transform(img)
    
    return img





     