import random

import argparse
import os
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from networks import Warper, l1_loss, tv_loss
from dataset import make_dataset
from utils import prepare_sub_folder, weights_init, str2bool, write_image

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data/WebCaricature_align_1.3_256')
parser.add_argument('--output_path', type=str, default='results/warper/')
parser.add_argument('--max_dataset_size', type=int, default=10000)

parser.add_argument('--resize_crop', type=str2bool, default=True)
parser.add_argument('--enlarge', type=str2bool, default=False)
parser.add_argument('--same_id', type=str2bool, default=True)
parser.add_argument('--hflip', type=str2bool, default=True)

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--iteration', type=int, default=20000)
parser.add_argument('--snapshot_log', type=int, default=100)
parser.add_argument('--snapshot_save', type=int, default=10000)

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0001)

parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--field_size', type=int, default=128)
parser.add_argument('--embedding_dim', type=int, default=32)
parser.add_argument('--warp_dim', type=int, default=64)
parser.add_argument('--scale', type=float, default=1.0)

parser.add_argument('--w_recon_img', type=float, default=10)
parser.add_argument('--w_recon_field', type=float, default=10)
parser.add_argument('--w_tv', type=float, default=0.000005)

args = parser.parse_args()

if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir, image_dir = prepare_sub_folder(args.output_path, delete_first=True)

    dataset = make_dataset(args)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                            num_workers=args.num_workers)

    warper = Warper(args)
    warper.to(device)
    warper.train()

    paras = list(warper.parameters())
    opt = optim.Adam([p for p in paras if p.requires_grad], lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-5)
    warper.apply(weights_init('kaiming'))

    train_iter = iter(dataloader)
    start = time.time()
    for step in range(0, args.iteration + 1):
        try:
            item = train_iter.next()
        except:
            train_iter = iter(dataloader)
            item = train_iter.next()

        if step > args.iteration // 2:
            opt.param_groups[0]['lr'] -= ((args.lr - 0.) / (args.iteration // 2))

        img_p = item['img_p'].to(device)
        img_c = item['img_c'].to(device)
        field_p2c = item['field_p2c'].to(device)
        field_m2c = item['field_m2c'].to(device)
        field_m2p = item['field_m2p'].to(device)

        opt.zero_grad()
        feat, embedding = warper.encode_p(img_p)
        img_recon = warper.decode_p(feat)
        loss_recon_p = l1_loss(img_p, img_recon) * args.w_recon_img

        z = warper.encode_f(field_m2c)
        _, field_recon = warper.decode_f(embedding, z, scale=args.scale)
        loss_recon_warp = l1_loss(field_recon, field_p2c) * args.w_recon_field

        random_z = torch.randn(img_p.size(0), args.warp_dim, 1, 1).cuda()
        _, field_gen = warper.decode_f(embedding, random_z, scale=args.scale)
        img_warp_gen = F.grid_sample(img_p, field_gen, align_corners=True)
        loss_tv = tv_loss(img_warp_gen) * args.w_tv

        loss_total = loss_recon_p + loss_recon_warp + loss_tv
        loss_total.backward()
        opt.step()

        # output log
        if (step + 1) % args.snapshot_log == 0:
            end = time.time()
            print('Step: {} ({:.0f}%) time:{} loss_rec_p:{:.4f} loss_rec_warp:{:.4f} loss_tv:{:.4f}'.format(
                step + 1,
                100.0 * step / args.iteration,
                int(end - start),
                loss_recon_p,
                loss_recon_warp,
                loss_tv))
            # input photo, input caricature, image_warp_p2c, image_warp_generated
            vis = torch.stack((img_p, img_c, F.grid_sample(img_p, field_p2c, align_corners=True), img_warp_gen), dim=1)
            write_image(step, image_dir, vis)

        # save checkpoint
        if (step + 1) % args.snapshot_save == 0:
            warper.save(checkpoint_dir, step)
