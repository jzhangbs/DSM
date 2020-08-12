import torch
import torch.utils.data as data
from typing import Callable, List, Tuple, Union, Iterable
import numpy as np
import json
import os


class FilelistDataset(data.Dataset):

    def __init__(self, root: str, list_file: str,
                 read: Callable[[List[str]], Iterable],
                 transforms: List[Callable[[Iterable], Iterable]]):

        with open(list_file, 'r') as f:
            fnl = json.load(f)
            for i in range(len(fnl)):
                for j in range(3):
                    if fnl[i][j] != '':
                        fnl[i][j] = os.path.join(root, fnl[i][j])
            self.filename_list = fnl
        self.root = root
        self.read = read
        self.transforms = transforms

    def __getitem__(self, index):
        images = self.read(self.filename_list[index])
        for T in self.transforms:
            images = T(images)
        return images

    def __len__(self):
        return len(self.filename_list)


def cycle(iterable: Iterable, num_cycle: int):
    for i in range(num_cycle):
        for it in iterable:
            yield it


def numpy_collate(data_: List[List[np.ndarray]]):
    transpose = zip(*data_)
    return [np.stack(batch, axis=0) for batch in transpose]
