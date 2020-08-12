import argparse
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
# from tensorboardX import SummaryWriter
import numpy as np

import sceneflow, kitti
from utils import Notify, info, fail, TimeMan
import model as model_unet, lightweight as model_lw, psm as model_psm
from io_utils import save_model, load_model


parser = argparse.ArgumentParser(description='stereo')

parser.add_argument('--num_worker', type=int, default=6)

parser.add_argument('--data_root', type=str, default=None)
parser.add_argument('--dataset', type=str, default='d,m,f')

parser.add_argument('--base', type=str, choices=['unet', 'lw', 'psm'], default='unet')

parser.add_argument('--max_d', type=int, default=192)
parser.add_argument('--crop_height', type=int, default=256)
parser.add_argument('--crop_width', type=int, default=512)

parser.add_argument('--lr', type=str, default='1e-3,.5e-3,.25e-3,.125e-3')
parser.add_argument('--boundaries', type=str, default='.625,.75,.875')
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--epoch', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=2)

parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--load_step', type=int, default=-1)
parser.add_argument('--reset_step', action='store_true', default=False)

parser.add_argument('--job_name', type=str, default='temp')

# parser.add_argument('--log_dir', type=str, default='log')
parser.add_argument('--save_dir', type=str, default='save')

parser.add_argument('--display', type=int, default=100)
parser.add_argument('--validation', type=int, default=-1)
parser.add_argument('--snapshot', type=int, default=2000)
parser.add_argument('--max_keep', type=int, default=1)

args = parser.parse_args()

if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True

    subsets = args.dataset.split(',')
    train_sceneflow = any([s in subsets for s in ['d', 'm', 'f']])
    train_kitti = any([s in subsets for s in ['k12', 'k15']])
    if train_kitti and train_sceneflow:
        raise Exception('Cannot train sceneflow and kitti together.')
    if train_sceneflow:
        get_train_loader = sceneflow.get_train_loader
    else:
        get_train_loader = kitti.get_train_loader
    dataset, loader = get_train_loader(args.data_root, subsets, args.epoch, args.batch_size,
                              {'crop_height': args.crop_height, 'crop_width': args.crop_width},
                              args.num_worker)

    step_per_epoch = math.ceil(len(dataset) / args.batch_size)
    total_step = step_per_epoch * args.epoch
    info(f'training sample: {len(dataset)}, step per epoch: {step_per_epoch}, total step: {total_step}')

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

    if args.load_path is None:
        global_step = 0
    else:
        global_step = load_model(model, args.load_path, args.load_step)
        if args.reset_step: global_step = 0
        info(f'load {args.load_path}')

    lr = [float(v) for v in args.lr.split(',')]
    boundaries = args.boundaries
    if boundaries is not None:
        boundaries = [int(total_step * float(b)) for b in boundaries.split(',')]
    optimizer = optim.Adam(model.parameters(), lr=lr[0], weight_decay=args.weight_decay)
    def piecewise_constant():
        if boundaries is None: return lr[0]
        i = 0
        for b in boundaries:
            if global_step < b: break
            i += 1
        curr_lr = lr[i]
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr
        return curr_lr
    model.train()

    time_man = TimeMan()
    time_man.start()
    # writer = SummaryWriter(os.path.join(args.log_dir, args.job_name))
    for left_image, right_image, disp_image in loader:
        if global_step >= total_step: break

        time_man.tic()
        curr_lr = piecewise_constant()
        left_image, right_image, disp_image = [torch.from_numpy(arr).cuda() for arr in
                                               [left_image, right_image, disp_image]]
        optimizer.zero_grad()
        output = model([left_image, right_image, disp_image])
        losses = compute_loss(output, disp_image)
        initial_loss, uncert_loss, loss, val_loss, less1, less3, d1 = losses
        if np.isnan(loss.item()):
            optimizer.zero_grad()
            fail(f'nan: {global_step}/{total_step}')
        loss.backward()
        optimizer.step()
        duration = time_man.toc()

        losses_np = [v.item() for v in losses]
        initial_loss, uncert_loss, loss, val_loss, less1, less3, d1 = losses_np

        # print
        if global_step % args.display == 0:
            remaining = time_man.remaining(total_step - global_step)
            end = time_man.end(total_step - global_step)
            info(f'step {global_step}/{total_step}, loss {loss:.4f} ({initial_loss:.4f} {uncert_loss:.4f} {val_loss:.4f}), (<1px) {less1:.4f}, (<3px) {less3:.4f} ({duration:.3f} sec/step, remaining {remaining} {end})')

        # write summary
        # if global_step % args.display == 0:
        #     log_iter = global_step * args.batch_size
        #     writer.add_scalar('loss/initial', initial_loss, log_iter)
        #     writer.add_scalar('loss/uncert', uncert_loss, log_iter)
        #     writer.add_scalar('loss/train', loss, log_iter)
        #     writer.add_scalar('loss/epe', val_loss, log_iter)
        #     writer.add_scalar('lr', curr_lr, log_iter)
        #     writer.add_scalar('less_one_accuracy/train', less1, log_iter)
        #     writer.add_scalar('less_three_accuracy/train', less3, log_iter)
        #     writer.add_histogram('uncert', output[1].clone().cpu().data.numpy(), log_iter)

        # save
        if global_step != 0 and global_step % args.snapshot == 0:
            save_model({
                'global_step': global_step,
                'state_dict': model.state_dict()
            }, args.save_dir, args.job_name, global_step, args.max_keep)

        global_step += 1

    save_model({
        'global_step': global_step,
        'state_dict': model.state_dict()
    }, args.save_dir, args.job_name, global_step, args.max_keep)
