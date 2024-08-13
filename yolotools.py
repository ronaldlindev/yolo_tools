import os
import cv2
import numpy as np
import matplotlib.pyplot as plt



# for working with yolo formatted data

def setup_yolo(ds_name):
    os.makedirs(ds_name, exist_ok=True)
    os.makedirs(os.path.join(ds_name, 'images'), exist_ok=True)
    os.makedirs(os.path.join(ds_name, 'labels'), exist_ok=True)

def normalize(raw, max):
    return raw /max

def to_xywh(x1, y1, x2, y2):
    x = (x1 + x2) / 2.0
    y = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return x, y, w, h

def to_xyxy(x, y, w, h):
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return x1, y1, x2, y2

    # takes xywh norm 
def plot(img: np.array, labels: list): 
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

    # Plot the image and boxes
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image_rgb)

    # Draw boxes
    x_center, y_center, width, height = labels
    # print(image.shape)
    img_height, img_width, _ = img.shape

    x_left = (x_center - (width/2)) * img_width
    y_top = (y_center - (height/2)) * img_height
    true_width = width * img_width
    true_height = height * img_height
    prev = (x_center, y_center, width, height)
    ax.set_title(f'{x_center}, {y_center}, {width}, {height}')
    rect = plt.Rectangle((x_left, y_top), true_width,
                        true_height, fill=False, color='red', linewidth=2)
    ax.add_patch(rect)

    plt.axis('off')
    plt.show(block=False)
    plt.pause(10.0)
    plt.close()

