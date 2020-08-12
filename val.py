import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import cv2

import sceneflow, kitti
import model as model_unet, lightweight as model_lw, psm as model_psm
from utils import info, TimeMan, fail
from io_utils import load_model


parser = argparse.ArgumentParser(description='stereo')

parser.add_argument('--data_root', type=str, default=None)
parser.add_argument('--dataset', type=str, default='f')

parser.add_argument('--base', type=str, choices=['unet', 'lw', 'psm'], default='unet')

parser.add_argument('--max_d', type=int, default=192)

parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--load_step', type=int, default=-1)

parser.add_argument('--write_result', action='store_true', default=False)
parser.add_argument('--result_dir', type=str, default=None)

parser.add_argument('--display', type=int, default=100)

args = parser.parse_args()

if __name__ == '__main__':

    subsets = args.dataset.split(',')
    train_sceneflow = any([s in subsets for s in ['d', 'm', 'f']])
    train_kitti = any([s in subsets for s in ['k12', 'k15']])
    if train_kitti and train_sceneflow:
        raise Exception('Cannot train sceneflow and kitti together.')
    if train_sceneflow:
        get_val_loader = sceneflow.get_val_loader
    else:
        get_val_loader = kitti.get_val_loader
    dataset, loader = get_val_loader(args.data_root, subsets, {})
    total_step = len(dataset)
    info(f'total step {total_step}')

    model_zoo = {
        'unet': model_unet,
        'lw': model_lw,
        'psm': model_psm
    }
    Model = model_zoo[args.base].Model
    Loss = model_zoo[args.base].Loss

    model = Model(args.max_d)
    model = nn.DataParallel(model)
    model.cuda()
    info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    compute_loss = Loss(args.max_d)

    load_model(model, args.load_path, args.load_step)

    model.eval()

    time_man = TimeMan()
    history = []
    with torch.no_grad():
        for global_step, (left_image, right_image, disp_image) in enumerate(loader):
            if args.write_result and global_step > 500: break
            time_man.tic()
            left_image, right_image, disp_image = [torch.from_numpy(arr).cuda() for arr in
                                                   [left_image, right_image, disp_image]]
            output = model([left_image, right_image, disp_image])
            losses = compute_loss(output, disp_image)
            # initial_loss, uncert_loss, loss, val_loss, less1, less3, d1 = losses
            duration = time_man.toc()

            output_np = [v.clone().cpu().data.numpy() for v in output]
            losses_np = [v.item() for v in losses]
            estimated_disp_image, uncertainty_image, refined_disp_image = output_np[-3:]
            initial_loss, uncert_loss, loss, val_loss, less1, less3, d1 = losses_np

            if global_step % args.display == 0:
                info(f'step {global_step}/{total_step}, val_loss {val_loss:.4f} ({initial_loss:.4f} {uncert_loss:.4f} {loss:.4f}), (<1px) {less1:.4f}, (<3px) {less3:.4f}, (D1) {d1:.4f} ({duration:.3f} sec/step)')

            if args.write_result:
                os.makedirs(args.result_dir, exist_ok=True)
                estimated_disp_image_uint16 = (estimated_disp_image * 256).astype(np.uint16)
                cv2.imwrite(os.path.join(args.result_dir, f'{global_step:06}_10e.png'),
                            estimated_disp_image_uint16[0].transpose((1, 2, 0)))
                refined_disp_image_uint16 = (refined_disp_image * 256).astype(np.uint16)
                cv2.imwrite(os.path.join(args.result_dir, f'{global_step:06}_10.png'),
                            refined_disp_image_uint16[0].transpose((1, 2, 0)))
                uncertainty_image = (uncertainty_image - np.min(uncertainty_image)) / (np.max(uncertainty_image) - np.min(uncertainty_image) + 1e-9) * 256
                uncertainty_image_uint16 = (uncertainty_image * 256).astype(np.uint16)
                cv2.imwrite(os.path.join(args.result_dir, f'{global_step:06}_10u.png'),
                            uncertainty_image_uint16[0].transpose((1, 2, 0)))

            if (not np.isnan(loss)) and (len(disp_image[disp_image <= 192.])/np.prod(disp_image.shape) >= .1):
                history.append(losses_np)
            else:
                fail(f'nan: {global_step}')

    avg_losses = [sum(h) / len(h) for h in zip(*history)]
    avg_initial_loss, avg_uncert_loss, avg_loss, avg_val_loss, avg_less1, avg_less3, avg_d1 = avg_losses
    info(f'average, val_loss {avg_val_loss:.4f} ({avg_initial_loss:.4f} {avg_uncert_loss:.4f} {avg_loss:.4f}), (<1px) {avg_less1:.4f}, (<3px) {avg_less3:.4f}, (D1) {avg_d1:.4f}')
