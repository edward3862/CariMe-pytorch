import random

import matplotlib.pyplot as plt
import argparse
import os
import torch
import torch.optim as optim
import tensorboardX
import numpy as np

from networks import l1_loss, RhoClipper
from networks.styler import Styler
from utils import prepare_sub_folder, unloader, weights_init, str2bool
from dataset import make_dataset
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data/WebCaricature_align_1.3_256')
parser.add_argument('--output_path', type=str, default='result/styler/')
parser.add_argument('--max_dataset_size', type=int, default=100000)

parser.add_argument('--resize_crop', type=str2bool, default=True)
parser.add_argument('--enlarge', type=str2bool, default=True)
parser.add_argument('--hflip', type=str2bool, default=True)
parser.add_argument('--same_id', type=str2bool, default=False)

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--iteration', type=int, default=500000)
parser.add_argument('--snapshot_log', type=int, default=100)
parser.add_argument('--snapshot_view', type=int, default=200)
parser.add_argument('--snapshot_save', type=int, default=100000)

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--style_dim', type=int, default=8)
parser.add_argument('--down_es', type=int, default=2)
parser.add_argument('--restype', type=str, default='adalin')
parser.add_argument('--w_recon_img', type=float, default=10)
parser.add_argument('--w_recon_s', type=float, default=1)
parser.add_argument('--w_recon_c', type=float, default=1)
args = parser.parse_args()

if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir, image_dir = prepare_sub_folder(args.output_path)
    train_writer = tensorboardX.SummaryWriter(os.path.join(args.output_path, "logs"))

    dataset = make_dataset(args)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                            num_workers=args.num_workers)

    model = Styler(args)
    model.to(device)
    model.train()

    gen_para = list(model.gen.parameters())
    dis_para = list(model.dis.parameters())
    gen_opt = optim.Adam([p for p in gen_para if p.requires_grad], lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-5)
    dis_opt = optim.Adam([p for p in dis_para if p.requires_grad], lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-5)
    model.apply(weights_init('kaiming'))
    rho_clipper = RhoClipper(0, 1)

    train_iter = iter(dataloader)
    for step in range(0, args.iteration + 1):
        try:
            item = train_iter.next()
        except:
            train_iter = iter(dataloader)
            item = train_iter.next()

        if step > (args.iteration // 2):
            gen_opt.param_groups[0]['lr'] -= ((args.lr - 0) / (args.iteration // 2))
            dis_opt.param_groups[0]['lr'] -= ((args.lr - 0) / (args.iteration // 2))

        img_p = item['img_p'].to(device)
        img_c = item['img_c'].to(device)

        # update discriminator
        dis_opt.zero_grad()
        random_s = torch.randn(img_p.size(0), args.style_dim, 1, 1).cuda()

        content_p, _ = model.gen.encode(img_p)
        fake_c = model.gen.decode(content_p, random_s)

        loss_adv_dis = model.dis.calc_dis_loss(img_c, fake_c)
        loss_adv_dis.backward()
        dis_opt.step()

        # update generator
        gen_opt.zero_grad()
        random_s = torch.randn(img_p.size(0), 8, 1, 1).cuda()

        content_p, style_p = model.gen.encode(img_p)
        content_c, style_c = model.gen.encode(img_c)

        recon_p = model.gen.decode(content_p, style_p)
        recon_c = model.gen.decode(content_c, style_c)

        img_p2c = model.gen.decode(content_p, random_s)
        recon_content_p, recon_s = model.gen.encode(img_p2c)

        loss_recon_img = (l1_loss(recon_c, img_c) + l1_loss(recon_p, img_p)) * args.w_recon_img
        loss_recon_style = l1_loss(recon_s, random_s) * args.w_recon_s
        loss_recon_content = l1_loss(recon_content_p, content_p) * args.w_recon_c
        loss_adv_gen = model.dis.calc_gen_loss(img_p2c)

        loss_gen = loss_recon_img + loss_adv_gen + loss_recon_content + loss_recon_style
        loss_gen.backward()
        gen_opt.step()

        model.gen.apply(rho_clipper)

        # output log
        if (step + 1) % args.snapshot_log == 0:
            losses = dict()
            losses['loss_recon_img'] = loss_recon_img
            losses['loss_recon_content'] = loss_recon_content
            losses['loss_recon_style'] = loss_recon_style
            losses['loss_adv_dis'] = loss_adv_dis
            losses['loss_adv_gen'] = loss_adv_gen
            losses['loss_gen'] = loss_gen

            train_writer.add_scalars('losses', losses, step + 1)

        # print training results
        if (step + 1) % args.snapshot_view == 0:
            print(
                'Step: {} ({:.0f}%)'.format(
                    step + 1,
                    100.0 * step / args.iteration,
                ))
            plt.figure()
            # input image
            plt.subplot(2, 3, 1)
            plt.title('input photo')
            plt.imshow(unloader(img_p[0].detach().cpu()))

            # input caricature
            plt.subplot(2, 3, 2)
            plt.title('input caricature')
            plt.imshow(unloader(img_c[0].detach().cpu()))

            # img_p2c
            plt.subplot(2, 3, 3)
            plt.title('img p2c')
            plt.imshow(unloader(img_p2c[0].detach().cpu()))

            # # recon image
            plt.subplot(2, 3, 4)
            plt.title('recon photo')
            plt.imshow(unloader(recon_p[0].detach().cpu()))

            # recon caricature
            plt.subplot(2, 3, 5)
            plt.title('recon caricature')
            plt.imshow(unloader(recon_c[0].detach().cpu()))

            plt.savefig(os.path.join(image_dir, 'step{}.jpg'.format(
                str(step + 1).zfill(6),
            )))

            plt.close()

        # save checkpoint
        if (step + 1) % args.snapshot_save == 0:
            model.save(checkpoint_dir, step)
