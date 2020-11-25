import random

import argparse
import os

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from networks import Warper
from utils import unloader, str2bool, shutil
from dataset import make_dataset
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data/WebCaricature_align_1.3_256')
parser.add_argument('--name', type=str, default='result/warper')
parser.add_argument('--model', type=str, default='warper_00020000.pt')
parser.add_argument('--output_dir', type=str, default='test')
parser.add_argument('--hflip', type=str2bool, default=False)
parser.add_argument('--enlarge', type=str2bool, default=False)
parser.add_argument('--resize_crop', type=str2bool, default=False)
parser.add_argument('--same_id', type=str2bool, default=True)

parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)

parser.add_argument('--psmap', type=int, default=128)
parser.add_argument('--embedding_dim', type=int, default=32)
parser.add_argument('--warp_dim', type=int, default=64)
parser.add_argument('--ups_dw', type=int, default=4)
parser.add_argument('--scale', type=float, default=1.0)

args = parser.parse_args()

if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(args.name, 'checkpoints', args.model)
    print('load model: ', model_path)
    output_path = os.path.join(args.name, args.output_dir)
    print('output path: ', output_path)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dataset = make_dataset(args)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers)

    warper = Warper(args)
    state_dict = torch.load(model_path)
    warper.load_state_dict(state_dict)
    warper.to(device)
    warper.eval()

    for batch, item in tqdm(enumerate(dataloader)):
        img_p = item['img_p'].to(device)
        names = item['name']
        filenames = item['filename']

        z = torch.randn(img_p.size()[0], args.warp_dim, 1, 1).cuda()
        img_warp, psmap, flows = warper(img_p, z, scale=args.scale)

        for i in range(img_p.size()[0]):
            input = img_p[i]
            result = img_warp[i]
            flow = flows[i]
            name = names[i]
            filename = filenames[i]

            output = torch.cat((input, result), dim=2)
            unloader(output.detach().cpu()).save(os.path.join(output_path, '{}_{}.jpg'.format(name, filename)), 'jpeg')
