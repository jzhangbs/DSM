import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

from psmnet.stackhourglass import PSMNet
from nn_utils import UNet, multi_dims


class UncertNet(nn.Module):

    def __init__(self):
        super(UncertNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        for m in self.conv1.modules():
            if any([isinstance(m, T) for T in [nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d]]):
                nn.init.xavier_uniform_(m.weight)
            elif any([isinstance(m, T) for T in [nn.BatchNorm2d, nn.BatchNorm3d]]):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        for m in self.conv2.modules():
            if any([isinstance(m, T) for T in [nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d]]):
                nn.init.xavier_uniform_(m.weight)
            elif any([isinstance(m, T) for T in [nn.BatchNorm2d, nn.BatchNorm3d]]):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
        self.conv3 = nn.Conv2d(32, 1, 3, 1, 1, bias=False)
        nn.init.xavier_uniform_(self.conv3.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        out = self.conv3(out)
        return out


class RefineNet(nn.Module):

    def __init__(self):
        super(RefineNet, self).__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(5, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.unet = UNet(32, 3, 2, 2, [], [32, 64], [], 'refine', 2)
        self.final_deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        for m in itertools.chain(self.init_conv.modules(), self.final_deconv.modules()):
            if any([isinstance(m, T) for T in [nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d]]):
                nn.init.xavier_uniform_(m.weight)
            elif any([isinstance(m, T) for T in [nn.BatchNorm2d, nn.BatchNorm3d]]):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.final_conv = nn.Conv2d(37, 1*8, 3, 1, 1, bias=False)
        nn.init.constant_(self.final_conv.weight, 0)

    def gen_kernel(self, input_):
        abs_sum = torch.sum(torch.abs(input_), dim=1, keepdim=True)
        input_ = input_ / (abs_sum + 1e-9)
        sum_ = torch.sum(input_, dim=1, keepdim=True)
        out = torch.cat([(1-sum_), input_], dim=1)
        out = out.contiguous()
        return out

    def forward(self, x):
        out = self.init_conv(x)
        out = self.unet(out)
        out = self.final_deconv(out)
        out = torch.cat([out, x], dim=1)
        out = self.final_conv(out)
        out = self.gen_kernel(out)
        return out


class Model(nn.Module):

    def __init__(self, max_d):
        super(Model, self).__init__()
        self.psm = PSMNet(max_d)
        self.uncert_net = UncertNet()
        self.refine_net = RefineNet()
        self.max_d = max_d

    def im2col(self, disp, radius):
        size = disp.size()
        offsets = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
        offsets = [[(i * k, j * k) for k in range(1, radius + 1)] for i, j in offsets]
        offsets = [(0, 0)] + sum(offsets, [])
        out = torch.cuda.FloatTensor(size[0], len(offsets), size[2], size[3]).zero_()
        for k, (i, j) in enumerate(offsets):
            out[:, k, max(0, i):min(size[2], size[2] + i), max(0, j):min(size[3], size[3] + j)] = \
                disp[:, 0, max(0, -i):min(size[2], size[2] - i), max(0, -j):min(size[3], size[3] - j)]
        out = out.contiguous()
        return out

    def forward(self, images):
        left, right, disp_true = images
        pred1, pred2, estimated_disp_image, prob_volume = self.psm(left, right)
        pred1, pred2, estimated_disp_image = [a.unsqueeze(1) for a in [pred1, pred2, estimated_disp_image]]

        prob_volume_detach = prob_volume
        estimated_disp_image_detach = estimated_disp_image

        entropy = torch.sum(-prob_volume * torch.log(torch.clamp(prob_volume_detach, 1e-9, 1.)), dim=1, keepdim=True)
        # log_variance = torch.log(torch.clamp(self.variance(prob_volume), min=1e-9))
        uncertainty_image = self.uncert_net(entropy)

        # center = estimated_disp_image.mean(dim=[1, 2, 3], keepdim=True)
        # scale = multi_dims(torch.std, estimated_disp_image, dim=[1, 2, 3], keepdim=True) + 1e-9
        # normalized_estimated_disp_image = (estimated_disp_image - center) / scale
        normalized_estimated_disp_image = estimated_disp_image_detach

        rgbdu = torch.cat([left, normalized_estimated_disp_image, entropy], dim=1)
        diff_kernel = self.refine_net(rgbdu)
        refined_disp_image = normalized_estimated_disp_image

        for _ in range(24):
            refined_disp_image = torch.sum(self.im2col(refined_disp_image, 1) * diff_kernel, dim=1, keepdim=True)
        # refined_disp_image += normalized_estimated_disp_image
        # refined_disp_image = refined_disp_image * scale + center

        # uncertainty_image = torch.cuda.FloatTensor(*estimated_disp_image.size()).zero_()
        # refined_disp_image = estimated_disp_image

        refined_disp_image = torch.clamp(refined_disp_image, 0., 256.)

        if refined_disp_image.size() != disp_true.size():
            final_size = disp_true.size()
            size = refined_disp_image.size()
            pred1, pred2, estimated_disp_image, refined_disp_image = \
                [image * final_size[3] / size[3]
                 for image in [pred1, pred2, estimated_disp_image, refined_disp_image]]
            pred1, pred2, estimated_disp_image, uncertainty_image, refined_disp_image = \
                [F.interpolate(image, size=final_size[2:], mode='bilinear')
                 for image in [pred1, pred2, estimated_disp_image, uncertainty_image, refined_disp_image]]
            # pred1, pred2, estimated_disp_image, uncertainty_image, refined_disp_image = \
            #     [image[..., size[2]-final_size[2]:, size[3]-final_size[3]:]
            #      for image in [pred1, pred2, estimated_disp_image, uncertainty_image, refined_disp_image]]

        return pred1, pred2, estimated_disp_image, uncertainty_image, refined_disp_image


class Loss(nn.Module):

    def __init__(self, max_d):
        super(Loss, self).__init__()
        self.max_d = max_d
        self.mask = None

    def l1(self, true, pred):
        abs_err = torch.abs(true[self.mask] - pred[self.mask])
        epe = torch.mean(abs_err)
        return epe

    def smooth_l1(self, true, pred):
        return nn.SmoothL1Loss()(true[self.mask], pred[self.mask])

    def uncertainty_loss(self, true, pred, uncert):
        abs_err = torch.abs(true[self.mask] - pred[self.mask])
        log_likelihood = abs_err * torch.exp(-uncert[self.mask]) + uncert[self.mask]
        # log_likelihood = abs_err * nn.Sigmoid()(-uncert[self.mask]) + torch.log(torch.exp(uncert[self.mask]) + 1)
        return torch.mean(log_likelihood)

    def refine_loss(self, true, pred, uncert):
        abs_err = torch.abs(true[self.mask] - pred[self.mask])
        scaled_loss = abs_err * (1 - nn.Sigmoid()(-uncert[self.mask]))
        return torch.mean(scaled_loss)

    def less(self, true, pred, thresh):
        abs_err = torch.abs(true[self.mask] - pred[self.mask])
        return torch.mean((abs_err < thresh).float())

    def kitti_d1(self, true, pred):
        abs_err = torch.abs(true[self.mask] - pred[self.mask])
        thresh_c = 3.
        thresh_p = true * .05
        return 100 * torch.mean(torch.min((abs_err > thresh_c).float(), (abs_err > thresh_p[self.mask]).float()))

    def forward(self, images, disp_true):

        pred1, pred2, estimated_disp_image, uncertainty_image, refined_disp_image = images

        self.mask = torch.min((disp_true <= self.max_d), (disp_true != 0))
        # self.mask = torch.min(self.mask, (torch.from_numpy(_uncertainty_image).cuda() >= 0.))
        self.mask.detach_()

        initial_loss = self.l1(disp_true, estimated_disp_image)
        uncert_loss = self.uncertainty_loss(disp_true, estimated_disp_image, uncertainty_image)
        val_loss = self.l1(disp_true, refined_disp_image)
        loss = self.smooth_l1(disp_true, estimated_disp_image) + \
               .5 * self.smooth_l1(disp_true, pred1) + \
               .7 * self.smooth_l1(disp_true, pred2) + \
               uncert_loss + val_loss
        less1 = self.less(disp_true, refined_disp_image, 1.)
        less3 = self.less(disp_true, refined_disp_image, 3.)
        d1 = self.kitti_d1(disp_true, refined_disp_image)

        return initial_loss, uncert_loss, loss, val_loss, less1, less3, d1