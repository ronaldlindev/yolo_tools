import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import ToTensor, Resize
from torchvision.transforms.functional import adjust_brightness, adjust_contrast

def _resize(img: Tensor, width: int = 640):
    return Resize((width, width))(img)

def _rebright(img: Tensor, bright_factor: float = 1.5):
    return adjust_brightness(img, bright_factor)

def _contrast(img: Tensor, contrast_factor = 1.5):
    return adjust_contrast(img, contrast_factor)

def _read_image(img_path: str):
    return ToTensor()(cv2.imread(img_path))

# has to be run with an event loop 
def do_transform(img_path: str):
    img = _read_image(img_path)
    img = _resize(img)
    img = _rebright(img)
    img = _contrast(img)
    # other transforms
      
    img = img.unsqueeze(0)

    if torch.cuda.is_available() and False: # retrain model with proper dimensions
        img = img.to('cuda')
    # convert to vram?? turn into tensor/??
    return img





     