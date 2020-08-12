import numpy as np
import random
import cv2
from typing import Callable, List, Tuple


def to_channel_first(images: List[np.ndarray]):
    return [np.transpose(img, [2, 0, 1]) for img in images]


def center_image(img: np.ndarray):
    """ normalize image """
    img = img.astype(np.float32)
    var = np.var(img, axis=(0, 1), keepdims=True)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 0.00000001)


def image_net_center(img: np.ndarray):
    stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    img = img.astype(np.float32)
    img /= 256.
    std = np.array([[stats['std'][::-1]]], dtype=np.float32)  # RGB to BGR
    mean = np.array([[stats['mean'][::-1]]], dtype=np.float32)
    return (img - mean) / (std + 0.00000001)


def random_crop(images: List[np.ndarray], height: int, width: int, seed_w: int=None, seed_h: int=None) -> List[np.ndarray]:
    left_image, right_image, disp_image = images
    h, w = left_image.shape[0:2]
    random.seed(seed_w)
    start_w = random.randint(0, w - width)
    random.seed(seed_h)
    start_h = random.randint(0, h - height)
    finish_w = start_w + width
    finish_h = start_h + height
    left_image_crop = left_image[start_h:finish_h, start_w:finish_w]
    right_image_crop = right_image[start_h:finish_h, start_w:finish_w]
    disp_image_crop = disp_image[start_h:finish_h, start_w:finish_w]

    return [left_image_crop, right_image_crop, disp_image_crop]


def resize(images: List[np.ndarray], height: int, width: int) -> List[np.ndarray]:
    left_image, right_image, disp_image = images
    left_image_resize = cv2.resize(left_image, (width, height), interpolation=cv2.INTER_CUBIC)
    right_image_resize = cv2.resize(right_image, (width, height), interpolation=cv2.INTER_CUBIC)

    return [left_image_resize, right_image_resize, disp_image]


def pad(images: List[np.ndarray], height: int, width: int) -> List[np.ndarray]:
    left_image, right_image, disp_image = images
    h, w = left_image.shape[0:2]
    dh, dw = height - h, width - w
    h_pad = (dh, 0)
    w_pad = (dw, 0)
    left_image_pad, right_image_pad = [np.pad(image, [h_pad, w_pad, (0, 0)], 'constant') for image in [left_image, right_image]]
    return [left_image_pad, right_image_pad, disp_image]


def identical(images: List[np.ndarray]) -> List[np.ndarray]:
    return images
