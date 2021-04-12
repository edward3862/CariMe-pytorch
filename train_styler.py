import random
import time

import argparse
import os
import torch
import torch.optim as optim
import numpy as np

from networks import Styler, l1_loss, RhoClipper
from utils import prepare_sub_folder, weights_init, str2bool, write_image
from dataset import make_dataset
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data/WebCaricature_align_1.3_256')
parser.add_argument('--output_path', type=str, default='results/styler/')
parser.add_argument('--max_dataset_size', type=int, default=100000)

parser.add_argument('--resize_crop', type=str2bool, default=True)
parser.add_argument('--enlarge', type=str2bool, default=True)
parser.add_argument('--same_id', type=str2bool, default=False)
parser.add_argument('--hflip', type=str2bool, default=True)

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--iteration', type=int, default=500000)
parser.add_argument('--snapshot_log', type=int, default=100)
parser.add_argument('--snapshot_vis', type=int, default=1000)
parser.add_argument('--snapshot_save', type=int, default=100000)

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--style_dim', type=int, default=8)
parser.add_argument('--w_recon_img', type=float, default=10)
parser.add_argument('--w_cyc_s', type=float, default=1)
parser.add_argument('--w_cyc_c', type=float, default=1)
args = parser.parse_args()

if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir, image_dir = prepare_sub_folder(args.output_path)

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
    start = time.time()
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

        content_p, _ = model.encode(img_p)
        fake_c = model.decode(content_p, random_s)

        loss_adv_dis = model.dis.calc_dis_loss(img_c, fake_c)
        loss_adv_dis.backward()
        dis_opt.step()

        # update generator
        gen_opt.zero_grad()
        random_s = torch.randn(img_p.size(0), 8, 1, 1).cuda()

        content_p, style_p = model.encode(img_p)
        content_c, style_c = model.encode(img_c)

        recon_p = model.decode(content_p, style_p)
        recon_c = model.decode(content_c, style_c)

        img_p2c = model.decode(content_p, random_s)
        recon_content_p, recon_s = model.encode(img_p2c)

        loss_recon_img = (l1_loss(recon_c, img_c) + l1_loss(recon_p, img_p)) * args.w_recon_img
        loss_cyc_style = l1_loss(recon_s, random_s) * args.w_cyc_s
        loss_cyc_content = l1_loss(recon_content_p, content_p) * args.w_cyc_c
        loss_adv_gen = model.dis.calc_gen_loss(img_p2c)

        loss_gen = loss_recon_img + loss_adv_gen + loss_cyc_content + loss_cyc_style
        loss_gen.backward()
        gen_opt.step()

        model.gen.apply(rho_clipper)

        # output log
        if (step + 1) % args.snapshot_log == 0:
            end = time.time()
            print(
                'Step: {} ({:.0f}%) time: {} loss_adv_g:{:.4f} loss_adv_d:{:.4f} loss_recon_img:{:.4f} loss_cyc_c:{:.4f} loss_cyc_s:{:.4f} '.format(
                    step + 1,
                    100.0 * step / args.iteration,
                    int(end - start),
                    loss_adv_gen,
                    loss_adv_dis,
                    loss_recon_img,
                    loss_cyc_content,
                    loss_cyc_style
                ))

        if (step + 1) % args.snapshot_vis == 0:
            # input photo, input caricature, recon photo, recon caricature, image translated
            vis = torch.stack((img_p, img_c, recon_p, recon_c, img_p2c), dim=1)
            write_image(step, image_dir, vis)

        # save checkpoint
        if (step + 1) % args.snapshot_save == 0:
            model.save(checkpoint_dir, step)
