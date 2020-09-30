import random
import shutil

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import make_dataset
from networks import Warper, Styler
from utils import unloader, str2bool

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()

parser.add_argument('--data_root', type=str, default='data/WebCaricature_align_1.3_256')
parser.add_argument('--model_path_warper', type=str, default='pretrained/warper.pt')
parser.add_argument('--model_path_styler', type=str, default='pretrained/styler_gen.pt')
parser.add_argument('--output_path', type=str, default='result/generated')

parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--hflip', type=str2bool, default=False)
parser.add_argument('--enlarge', type=str2bool, default=False)
parser.add_argument('--resize_crop', type=str2bool, default=False)
parser.add_argument('--same_id', type=str2bool, default=True)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=8)

parser.add_argument('--psmap', type=int, default=128)
parser.add_argument('--embedding_dim', type=int, default=32)
parser.add_argument('--warp_dim', type=int, default=64)
parser.add_argument('--ups_dw', type=int, default=4)
parser.add_argument('--last_bn', type=str2bool, default=True)
parser.add_argument('--scale', type=float, default=1)

parser.add_argument('--style_dim', type=int, default=8)
parser.add_argument('--down_es', type=int, default=2)
parser.add_argument('--restype', type=str, default='adalin')

args = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def load_img(path):
    img = Image.open(path)
    img = transform(img)
    img = img.unsqueeze(0)
    return img


if __name__ == '__main__':

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('load warper: ', args.model_path_warper)
    print('load styler: ', args.model_path_styler)
    output_path = args.output_path
    print('output path: ', output_path)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    dataset = make_dataset(args)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers)

    warper = Warper(args)
    warper.load(args.model_path_warper)
    warper.to(device)
    warper.eval()

    styler = Styler(args)
    styler.load(args.model_path_styler)
    styler.to(device)
    styler.eval()

    num = 3

    for batch, item in tqdm(enumerate(dataloader)):
        img_p = item['img_p'].to(device)
        names = item['name']
        filenames = item['filename']

        results = []

        for i in range(num):
            z = torch.randn(img_p.size()[0], args.warp_dim, 1, 1).cuda()
            s = torch.randn(img_p.size()[0], args.style_dim, 1, 1).cuda()

            img_warp, psmap, _ = warper(img_p, z, scale=args.scale)
            img_style = styler(img_p, s)
            img_warp_style = styler(img_warp, s)

            results.append(img_warp_style.unsqueeze(0))
        results = torch.cat(results, dim=0).detach().cpu()

        for i in range(img_p.size()[0]):
            input = img_p[i].detach().cpu()
            name = names[i]
            filename = filenames[i]
            result = results[:, i, :, :, :]

            result = result.permute(1, 2, 0, 3)
            result = result.reshape(3, 256, 256 * num)

            output = torch.cat((input, result), dim=2)
            unloader(output).save(os.path.join(output_path, '{}_{}.jpg'.format(name, filename)), 'jpeg')

