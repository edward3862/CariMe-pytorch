import matplotlib.pyplot as plt
import argparse
import os
import shutil
import torch
import numpy as np
import random

from tqdm import tqdm
from networks.styler import Styler
from utils import unloader, str2bool
from dataset import make_dataset
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data/WebCaricature_align_1.3_256')
parser.add_argument('--name', type=str, default='result/styler')
parser.add_argument('--model', type=str, default='gen_00200000.pt')
parser.add_argument('--output_dir', type=str, default='test')

parser.add_argument('--resize_crop', type=str2bool, default=False)
parser.add_argument('--hflip', type=str2bool, default=False)
parser.add_argument('--enlarge', type=str2bool, default=False)
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--style_dim', type=int, default=8)
parser.add_argument('--down_es', type=int, default=2)
parser.add_argument('--restype', type=str, default='adalin')

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
    print(len(dataset))
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers)

    model = Styler(args)
    model.load(model_path)
    model.to(device)
    model.eval()

    for batch, item in tqdm(enumerate(dataloader)):

        img_ps = item['img_p'].to(device)
        names = item['name']
        filenames = item['filename']

        s = torch.randn(img_ps.size(0), 8, 1, 1).cuda()
        outputs = model(img_ps, s)

        for i in range(img_ps.size()[0]):
            input = img_ps[i].detach().cpu()
            output = outputs[i].detach().cpu()
            name = names[i]
            filename = filenames[i]

            figure = torch.cat((input, output), dim=2)
            unloader(figure).save(os.path.join(output_path, '{}_{}.jpg'.format(name, filename)), 'jpeg')