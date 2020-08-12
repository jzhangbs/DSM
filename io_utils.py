import numpy as np
import re
import os
import json
import torch
import torch.nn as nn
import sys


def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    endian = None
    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(br'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = data[::-1, ...]  # cv2.flip(data, 0)
    return data


def write_pfm(file, image, scale=1):
    file = open(file, 'wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)

    image.tofile(file)


def save_model(obj, save_dir: str, job_name: str, global_step: int, max_keep: int):
    os.makedirs(os.path.join(save_dir, job_name), exist_ok=True)
    record_file = os.path.join(save_dir, job_name, 'record')
    cktp_file = os.path.join(save_dir, job_name, f'{global_step}.tar')
    if not os.path.exists(record_file):
        with open(record_file, 'w+') as f:
            json.dump([], f)
    with open(record_file, 'r') as f:
        record = json.load(f)
    record.append(global_step)
    if len(record) > max_keep:
        old = record[0]
        record = record[1:]
        os.remove(os.path.join(save_dir, job_name, f'{old}.tar'))
    torch.save(obj, cktp_file)
    with open(record_file, 'w') as f:
        json.dump(record, f)


def load_model(model: nn.Module, load_path: str, load_step: int):
    if load_step is None:
        model.load_state_dict(torch.load(load_path)['state_dict'])
        return 0
    else:
        if load_step == -1:
            record_file = os.path.join(load_path, 'record')
            with open(record_file, 'r') as f:
                record = json.load(f)
            if len(record) == 0:
                raise Exception('no latest model.')
            load_step = record[-1]
        cktp_file = os.path.join(load_path, f'{load_step}.tar')
        model.load_state_dict(torch.load(cktp_file)['state_dict'])
        return torch.load(cktp_file)['global_step']
