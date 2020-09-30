import argparse
import math
import os
import shutil
import numpy as np

import torch
import torch.nn.init as init

from PIL import Image
from skimage import transform as sktf
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def load_filenames(data_root, name, token, file_type):
    files = os.listdir(os.path.join(data_root, token, name))
    return [os.path.join(data_root, token, name, file) for file in files if file.startswith(file_type)]


def make_position_map(length):
    temp_height = torch.linspace(start=-1.0, end=1.0, steps=length, requires_grad=False).view(length, 1)
    temp_width = torch.linspace(start=-1, end=1, steps=length, requires_grad=False).view(1, length)
    pos_x = temp_height.repeat(1, length).view(length, length, 1)
    pos_y = temp_width.repeat(length, 1).view(length, length, 1)
    return pos_y.numpy(), pos_x.numpy()


def make_constant_map(length=256):
    y, x = make_position_map(length)
    map = np.concatenate((y, x), axis=2)
    map = torch.from_numpy(map).unsqueeze(0)
    return map


def warp_image(image, src_points, dst_points):
    src_points = np.array(
        [
            [0, 0], [0, image.shape[0]],
            [image.shape[0], 0], list(image.shape[:2])
        ] + src_points.tolist()
    )
    dst_points = np.array(
        [
            [0, 0], [0, image.shape[0]],
            [image.shape[0], 0], list(image.shape[:2])
        ] + dst_points.tolist()
    )

    tform3 = sktf.PiecewiseAffineTransform()
    tform3.estimate(dst_points, src_points)

    warped = sktf.warp(image, tform3, output_shape=image.shape)
    return warped


def warp_position_map(p, c, length=256):
    pos_ys, pos_xs = make_position_map(length)
    pos_ys_warped = warp_image(pos_ys, p, c)
    pos_xs_warped = warp_image(pos_xs, p, c)
    return pos_ys_warped, pos_xs_warped


def unloader(img):
    img = (img + 1) / 2
    tf = transforms.Compose([
        transforms.ToPILImage()
    ])
    return tf(img)


def prepare_sub_folder(output_path, delete=True):
    print('preparing sub folder for {}'.format(output_path))
    if delete and os.path.exists(output_path):
        shutil.rmtree(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    checkpoint_path = os.path.join(output_path, 'checkpoints')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    images_path = os.path.join(output_path, 'images')
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    logs_path = os.path.join(output_path, 'logs')
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    return checkpoint_path, images_path


def weights_init(init_type='gaussian', mean=0.0, std=0.02):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, mean, std)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def load_img(path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    )
    img = Image.open(path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    return img


def cal_delta(psmap):
    pos_y, pos_x = make_position_map(256)
    const = np.concatenate((pos_y, pos_x), axis=2)
    const = torch.from_numpy(const).cuda()
    y1, x1 = const[:, :, 0], psmap[:, :, 1]
    y2, x2 = const[:, :, 0], psmap[:, :, 1]
    return torch.mean(torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

