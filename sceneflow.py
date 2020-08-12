import torch
import torch.utils.data as data
from typing import List, Tuple, Dict, Callable, Any, Iterable, Union
import numpy as np
import cv2

from io_utils import load_pfm
from imgproc import image_net_center as center_image, random_crop, resize, pad, to_channel_first
from data_utils import FilelistDataset, cycle, numpy_collate


def sceneflow_read(files: List[str]) -> List[np.ndarray]:
    l, r, d = files

    left_image = center_image(cv2.imread(l))
    right_image = center_image(cv2.imread(r))

    with open(d, 'rb') as df:
        disp_image = load_pfm(df)
    disp_image = np.expand_dims(disp_image, axis=-1)

    return [left_image, right_image, disp_image]


def get_train_loader(
        root: str,
        subsets: List[str],
        epoch: int,
        batch_size: int,
        preproc_args: Dict[str, Any],
        num_workers: int=0) -> Tuple[data.Dataset, Iterable]:

    lists = {
        'f': 'list/flyingthings3d_train.json',
        'm': 'list/monkaa_train.json',
        'd': 'list/driving_train.json'
    }
    datasets = []
    for s in subsets:
        d = FilelistDataset(root, lists[s], sceneflow_read, [
            lambda images: random_crop(images, preproc_args['crop_height'], preproc_args['crop_width']),
            to_channel_first
        ])
        datasets.append(d)
    dataset = datasets[0]
    for d in datasets[1:]:
        dataset += d
    loader = data.DataLoader(dataset, batch_size, collate_fn=numpy_collate, shuffle=True, num_workers=num_workers, drop_last=True)
    multi_epoch_loader = iter(cycle(loader, epoch))
    return dataset, multi_epoch_loader


def get_val_loader(
        root: str,
        subsets: List[str],
        preproc_args: Dict[str, Any]) -> Tuple[data.Dataset, Iterable]:

    lists = {
        'f': 'list/flyingthings3d_test.json',
        'm': 'list/monkaa_val.json',
        'd': 'list/driving_val.json'
    }
    datasets = []
    def resize_wrapper(images):
        h, w = images[0].shape[0:2]
        crop_width = w - w % 32 + 32
        crop_height = h - h % 32 + 32
        return resize(images, crop_height, crop_width)
    def pad_wrapper(images):
        h, w = images[0].shape[0:2]
        crop_width = w - w % 32 + 32
        crop_height = h - h % 32 + 32
        return pad(images, crop_height, crop_width)
    for s in subsets:
        d = FilelistDataset(root, lists[s], sceneflow_read, [
            resize_wrapper,
            to_channel_first
        ])
        datasets.append(d)
    dataset = datasets[0]
    for d in datasets[1:]:
        dataset += d
    return dataset, data.DataLoader(dataset, collate_fn=numpy_collate)
