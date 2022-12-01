import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal


def loss_int(fusion_image, images_ir, images_vi):
    g_loss_int = torch.mean(torch.square(fusion_image - images_ir)) + \
                5*torch.mean(torch.square(fusion_image - images_vi))

    return g_loss_int


def loss_grad(fusion_image, images_ir, images_vi):
    Image_vi_grad_lowpass = torch.abs(gradient(low_pass(images_vi)))
    Image_ir_grad_lowpass = torch.abs(gradient(low_pass(images_ir)))

    Image_vi_weight_lowpass = Image_vi_grad_lowpass
    Image_ir_weight_lowpass = Image_ir_grad_lowpass
    Image_vi_score_1 = 1

    Image_vi_score_2 = torch.sign(
        Image_vi_weight_lowpass - torch.minimum(Image_vi_weight_lowpass, Image_ir_weight_lowpass))

    Image_vi_score = Image_vi_score_1 * Image_vi_score_2
    Image_ir_score = 1 - Image_vi_score
    g_loss_grad = torch.mean(Image_ir_score * torch.square(gradient(fusion_image) - gradient(images_ir))) + \
                  torch.mean(Image_vi_score * torch.square(gradient(fusion_image) - gradient(images_vi)))

    return g_loss_grad


def gradient(image):
    _, _, h, w = image.shape
    # print(h, w)
    img = image
    k = torch.tensor([0., 1., 0., 1., -4., 1., 0., 1., 0.], dtype=torch.float)
    k = k.view(1, 1, 3, 3).cuda()
    # print(k.size(), k)
    z = F.conv2d(img, k, padding=1)
    result = z
    # print(result.shape)
    return result


def low_pass(image):
    _, _, h, w = image.shape
    # print(h, w)
    img = image
    k = torch.tensor([0.0947, 0.1183, 0.0947, 0.1183, 0.1478, 0.1183, 0.0947, 0.1183, 0.0947], dtype=torch.float)
    k = k.view(1, 1, 3, 3).cuda()
    # print(k.size(), k)
    z = F.conv2d(img, k, padding=1)
    result = z
    # print(result.shape)
    return result


def loss_de(sept_ir, sept_vi, images_ir, images_vi):
    g_loss_sept = torch.mean(torch.square(sept_ir - images_ir)) + torch.mean(torch.square(sept_vi - images_vi))
    return g_loss_sept


def loss_total(fusion_image, sept_ir, sept_vi, images_ir, images_vi):
    g_loss = 10 * loss_int(fusion_image, images_ir, images_vi) + \
             1 * loss_grad(fusion_image, images_ir, images_vi) + \
             10 * loss_de(sept_ir, sept_vi, images_ir, images_vi)
    return g_loss


if __name__ == '__main__':
    x1 = torch.randn(1, 1, 256, 256).cuda()
    x2 = torch.randn(1, 1, 256, 256).cuda()
    x3 = torch.randn(1, 1, 256, 256).cuda()
    x4 = torch.randn(1, 1, 256, 256).cuda()
    x5 = torch.randn(1, 1, 256, 256).cuda()
    # z = loss_de(x1, x2, x3, x4)
    # print(z)

    z = loss_total(x1, x2, x3, x4, x5)
    print(z)
