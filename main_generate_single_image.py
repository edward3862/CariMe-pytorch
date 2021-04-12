import random
import shutil

import argparse
import os
import torch
import numpy as np
from networks import Warper, Styler
from utils import load_img, unload_img

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()

parser.add_argument('--input_path', type=str, default='images/Meg Ryan/P00015.jpg')
parser.add_argument('--model_path_warper', type=str, default='results/warper/checkpoints/warper_00020000.pt')
parser.add_argument('--model_path_styler', type=str, default='results/styler/checkpoints/gen_00200000.pt')

parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--field_size', type=int, default=128)
parser.add_argument('--embedding_dim', type=int, default=32)
parser.add_argument('--warp_dim', type=int, default=64)
parser.add_argument('--style_dim', type=int, default=8)
parser.add_argument('--scale', type=float, default=1)
parser.add_argument('--generate_num', type=int, default=5)

args = parser.parse_args()


if __name__ == '__main__':
    output_path = os.path.join(args.input_path[:-4] + '_gen.jpg')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('load warper: ', args.model_path_warper)
    print('load styler: ', args.model_path_styler)

    warper = Warper(args)
    warper.load(args.model_path_warper)
    warper.to(device)
    warper.eval()

    styler = Styler(args)
    styler.load(args.model_path_styler)
    styler.to(device)
    styler.eval()

    num = args.generate_num
    img_p = load_img(args.input_path).to(device)
    results = []
    for i in range(num):
        z = torch.randn(img_p.size()[0], args.warp_dim, 1, 1).cuda()
        img_warp, psmap, _ = warper(img_p, z, scale=args.scale)

        s = torch.randn(img_p.size()[0], args.style_dim, 1, 1).cuda()
        img_style = styler(img_p, s)
        img_warp_style = styler(img_warp, s)

        results.append(img_warp_style)

    results = torch.cat(results, dim=3)
    output = torch.cat([img_p, results], dim=3).squeeze().detach().cpu()
    unload_img(output).save(output_path, 'jpeg')
