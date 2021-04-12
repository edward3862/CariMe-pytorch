import random

import argparse
import os

import torch
import numpy as np
from tqdm import tqdm

from networks import Warper
from utils import str2bool
from dataset import make_dataset
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data/WebCaricature_align_1.3_256')
parser.add_argument('--name', type=str, default='results/warper')
parser.add_argument('--model', type=str, default='warper_00020000.pt')

parser.add_argument('--resize_crop', type=str2bool, default=False)
parser.add_argument('--enlarge', type=str2bool, default=False)
parser.add_argument('--same_id', type=str2bool, default=False)
parser.add_argument('--hflip', type=str2bool, default=False)

parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=8)

parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--field_size', type=int, default=128)
parser.add_argument('--embedding_dim', type=int, default=32)
parser.add_argument('--warp_dim', type=int, default=64)
parser.add_argument('--scale', type=float, default=1.0)

args = parser.parse_args()


def make_field(length):
    temp_height = np.linspace(-1.0, 1.0, num=length).reshape(length, 1, 1)
    temp_width = np.linspace(-1.0, 1.0, num=length).reshape(1, length, 1)

    pos_x = np.repeat(temp_height, length, axis=1)
    pos_y = np.repeat(temp_width, length, axis=0)

    return np.concatenate((pos_y, pos_x), axis=2)


def cal_delta(map1, map2):
    y1, x1 = map1[:, :, 0], map1[:, :, 1]
    y2, x2 = map2[:, :, 0], map2[:, :, 1]
    return np.mean(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    print(args.name)
    print(args.scale)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(args.name, 'checkpoints', args.model)
    print('load model: ', model_path)

    dataset = make_dataset(args)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers)

    warper = Warper(args)
    state_dict = torch.load(model_path)
    warper.load_state_dict(state_dict)
    warper.to(device)
    warper.eval()

    deltas = []
    const = make_field(256)

    for batch, item in tqdm(enumerate(dataloader)):
        img_p = item['img_p'].to(device)
        names = item['name']
        filenames = item['filename']

        z = torch.randn(img_p.size()[0], args.warp_dim, 1, 1).cuda()
        _, fields, _ = warper(img_p, z, scale=args.scale)

        for i in range(img_p.size()[0]):
            field = fields[i].detach().cpu().numpy()
            deltas.append(cal_delta(const, field) * 256)
    print(np.mean(deltas))
    print(np.std(deltas))


