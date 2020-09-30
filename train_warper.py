import random

import argparse
import os

import tensorboardX
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from networks import l1_loss, tv_loss
from networks import Warper
from utils import unloader, prepare_sub_folder, weights_init, str2bool, cal_delta
from dataset import make_dataset
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data/WebCaricature_align_1.3_256')
parser.add_argument('--output_path', type=str, default='result/warper/')
parser.add_argument('--max_dataset_size', type=int, default=10000)

parser.add_argument('--resize_crop', type=str2bool, default=True)
parser.add_argument('--enlarge', type=str2bool, default=False)
parser.add_argument('--hflip', type=str2bool, default=True)
parser.add_argument('--same_id', type=str2bool, default=True)

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--iteration', type=int, default=20000)
parser.add_argument('--snapshot_log', type=int, default=50)
parser.add_argument('--snapshot_view', type=int, default=200)
parser.add_argument('--snapshot_save', type=int, default=10000)

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0001)

parser.add_argument('--psmap', type=int, default=128)
parser.add_argument('--embedding_dim', type=int, default=32)
parser.add_argument('--warp_dim', type=int, default=64)
parser.add_argument('--ups_dw', type=int, default=4)
parser.add_argument('--last_bn', type=str2bool, default=True)
parser.add_argument('--scale', type=float, default=1.0)

parser.add_argument('--w_recon_img', type=float, default=10)
parser.add_argument('--w_recon_psmap', type=float, default=10)
parser.add_argument('--w_tv', type=float, default=0.000005)

args = parser.parse_args()


if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir, image_dir = prepare_sub_folder(args.output_path, delete=True)
    train_writer = tensorboardX.SummaryWriter(os.path.join(args.output_path, "logs"))

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
        psmap_p2c = item['psmap_p2c'].to(device)
        psmap_m2c = item['psmap_m2c'].to(device)
        psmap_m2p = item['psmap_m2p'].to(device)
        ldmark_p = item['ldmark_p']
        ldmark_c = item['ldmark_c']

        opt.zero_grad()
        random_z = torch.randn(img_p.size(0), args.warp_dim, 1, 1).cuda()

        feat, embedding = warper.enc(img_p)
        img_recon = warper.dec(feat)

        z = warper.encode(psmap_m2c)
        _, psmap_recon = warper.decode(embedding, z, scale=args.scale)
        img_warp_recon = F.grid_sample(img_p, psmap_recon, align_corners=True)

        _, psmap_gen = warper.decode(embedding, random_z, scale=args.scale)
        img_warp_gen = F.grid_sample(img_p, psmap_gen, align_corners=True)
        psmap_m2c_recon = F.grid_sample(psmap_m2p.permute(0, 3, 1, 2), psmap_gen, align_corners=True).permute(0, 2, 3, 1)
        img_m2c_recon = F.grid_sample(img_p, psmap_m2c_recon, align_corners=True)

        loss_recon_img = l1_loss(img_p, img_recon) * args.w_recon_img
        loss_recon_psmap = l1_loss(psmap_recon, psmap_p2c) * args.w_recon_psmap
        loss_tv = tv_loss(img_warp_gen) * args.w_tv

        loss_total = loss_recon_img + loss_recon_psmap + loss_tv
        loss_total.backward()
        opt.step()

        # output log
        if (step + 1) % args.snapshot_log == 0:
            losses = dict()
            losses['loss_recon_img'] = loss_recon_img
            losses['loss_recon_psmap'] = loss_recon_psmap
            losses['loss_tv'] = loss_tv

            stat = dict()
            stat['mean_z'] = z.mean()
            stat['std_z'] = z.std()
            stat['mean_embedding'] = embedding.mean()
            stat['std_embedding'] = embedding.std()
            stat['delta'] = cal_delta(psmap_gen[0])

            train_writer.add_scalars('losses', losses, step + 1)
            train_writer.add_scalars('stat', stat, step + 1)

        # print training results
        if (step + 1) % args.snapshot_view == 0:
            print(
                'Step: {} ({:.0f}%)'.format(
                    step + 1,
                    100.0 * step / args.iteration
                ))

            plt.figure()

            # input image
            plt.subplot(3, 3, 1)
            plt.title('input photo')
            plt.imshow(unloader(img_p[0].detach().cpu()))
            plt.scatter(ldmark_p[0][:, 0], ldmark_p[0][:, 1], s=5)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')

            # recon image(ae)
            plt.subplot(3, 3, 2)
            plt.title('recon photo(ae)')
            plt.imshow(unloader(img_recon[0].detach().cpu()))
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')

            # input caricature
            plt.subplot(3, 3, 3)
            plt.title('input caricature')
            plt.imshow(unloader(img_c[0].detach().cpu()))
            plt.scatter(ldmark_c[0][:, 0], ldmark_c[0][:, 1], s=5)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')

            # img_p + warp_gt
            plt.subplot(3, 3, 4)
            plt.title('p + warp_p2c')
            plt.imshow(unloader(F.grid_sample(img_p[0].unsqueeze(0), psmap_p2c[0].unsqueeze(0), align_corners=True)[0].detach().cpu()))
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')

            # img_p + warp_recon
            plt.subplot(3, 3, 5)
            plt.title('p + warp_recon')
            plt.imshow(unloader(img_warp_recon[0].detach().cpu()))
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')

            # img_p + warp_gt
            plt.subplot(3, 3, 6)
            plt.title('p + warp_m2c')
            plt.imshow(unloader(F.grid_sample(img_p[0].unsqueeze(0), psmap_m2c[0].unsqueeze(0), align_corners=True)[0].detach().cpu()))
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')

            # img_p + m2p warp
            plt.subplot(3, 3, 7)
            plt.title('p + warp_m2p')
            plt.imshow(unloader(F.grid_sample(img_p[0].unsqueeze(0), psmap_m2p[0].unsqueeze(0), align_corners=True)[0].detach().cpu()))
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')

            # img_p + random warp
            plt.subplot(3, 3, 8)
            plt.title('p + warp(gen)')
            plt.imshow(unloader(img_warp_gen[0].detach().cpu()))
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')

            # img_p + warp_m2c_recon
            plt.subplot(3, 3, 9)
            plt.title('p + warp_m2c recon')
            plt.imshow(unloader(img_m2c_recon[0].detach().cpu()))
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')

            plt.savefig(os.path.join(image_dir, 'step{}.jpg'.format(
                str(step + 1).zfill(6),
            )))

            plt.close()

        # save checkpoint
        if (step + 1) % args.snapshot_save == 0:
            warper.save(checkpoint_dir, step)