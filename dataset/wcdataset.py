import os
import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms.functional as transF
from PIL import Image
import torchvision.transforms as transforms

from utils import make_constant_map, load_filenames, warp_position_map

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

flip_index = [0, 3, 2, 1, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 16, 15, 14]


class WCDataSet(data.Dataset):
    def __init__(self, args, name_list, mode='train', transform=transform):
        super(WCDataSet, self).__init__()
        self.data_root = args.data_root
        self.name_list = name_list
        self.mode = mode
        self.enlarge = args.enlarge
        self.hflip = args.hflip
        self.resize_crop = args.resize_crop
        self.same_id = args.same_id
        self.transform = transform
        self.const_map = make_constant_map().squeeze(0)

        self.p_dir = {name: load_filenames(self.data_root, name, 'image', 'P') for name in name_list}
        self.c_dir = {name: load_filenames(self.data_root, name, 'image', 'C') for name in name_list}

        self.num_p = sum([len(files) for files in self.p_dir.values()])
        self.num_c = sum([len(files) for files in self.c_dir.values()])

        print('load dataset over')
        print('{} image: {}, {} caricature: {}'.format(self.mode, self.num_p, self.mode, self.num_c))

        self.img_list = []
        for name in name_list:
            self.img_list += self.p_dir[name]
        assert len(self.img_list) == self.num_p

        if self.mode == 'train':
            self.size = min(args.max_dataset_size, 100000)
            self.ldmark_mean = torch.zeros(17, 2)
            for img_path in self.img_list:
                ldmark_path = img_path.replace('image', 'landmark').replace('.jpg', '.txt')
                ldmark = self._load_ldmark(ldmark_path)
                self.ldmark_mean += ldmark
            self.ldmark_mean /= self.num_p
        else:
            self.size = self.num_p

    def _sample_pair(self, same_id=True):

        name1 = random.choice(self.name_list)
        img_p_path = random.choice(self.p_dir[name1])
        ldmark_p_path = img_p_path.replace('image', 'landmark').replace('.jpg', '.txt')

        if same_id:
            img_c_path = random.choice(self.c_dir[name1])
        else:
            name2 = random.choice(self.name_list)
            img_c_path = random.choice(self.c_dir[name2])
        ldmark_c_path = img_c_path.replace('image', 'landmark').replace('.jpg', '.txt')
        return img_p_path, img_c_path, ldmark_p_path, ldmark_c_path

    def _load_img(self, path):
        return Image.open(path).convert('RGB')

    def _load_ldmark(self, path):
        return torch.from_numpy(np.loadtxt(path, delimiter='\t')).float()

    def _cal_psmap(self, src, dst):
        psmap_y, psmap_x = warp_position_map(src, dst)
        psmap = np.concatenate((psmap_y, psmap_x), axis=2)
        psmap = torch.from_numpy(psmap).float()
        return psmap

    def _random_horizonal_flip(self, img, ldmark):
        if random.random() < 0.5:
            img_hflip = transF.hflip(img)
            ldmark[:, 0] = 256 - ldmark[:, 0]
            ldmark_hflip = torch.zeros_like(ldmark)
            for i in range(ldmark.shape[0]):
                ldmark_hflip[i] = ldmark[flip_index[i]]
            return img_hflip, ldmark_hflip
        else:
            return img, ldmark

    def _random_enlarge_ldmark(self, ldmark, p=0.5):
        if random.random() < p:
            rate = random.random() / 5 + 1
        else:
            rate = 1
        # ldmark = (ldmark - self.ldmark_mean) * rate + self.ldmark_mean
        ldmark = (ldmark - self.ldmark_mean) * (rate - 1) + ldmark
        return ldmark, rate

    def _random_resize_crop(self, img, ldmark, resize=288):
        if random.random() < 0.5:
            w, h = img.size
            time = resize / w
            img = img.resize((resize, resize), Image.BILINEAR)
            x = random.random() * (time - 1) * w
            y = random.random() * (time - 1) * h
            img_crop = img.crop((x, y, x + w, y + h))
            ldmark_crop = ldmark * time
            ldmark_crop[:, 0] = ldmark_crop[:, 0] - x
            ldmark_crop[:, 1] = ldmark_crop[:, 1] - y
            return img_crop, ldmark_crop
        else:
            return img, ldmark

    def __getitem__(self, index):
        if self.mode == 'train':
            img_p_path, img_c_path, ldmark_p_path, ldmark_c_path = self._sample_pair(same_id=self.same_id)

            img_p = self._load_img(img_p_path)
            img_c = self._load_img(img_c_path)
            ldmark_p = self._load_ldmark(ldmark_p_path)
            ldmark_c = self._load_ldmark(ldmark_c_path)

            if self.hflip:
                img_p, ldmark_p = self._random_horizonal_flip(img_p, ldmark_p)
                img_c, ldmark_c = self._random_horizonal_flip(img_c, ldmark_c)

            if self.resize_crop:
                img_p, ldmark_p = self._random_resize_crop(img_p, ldmark_p)
                img_c, ldmark_c = self._random_resize_crop(img_c, ldmark_c)

            if self.enlarge:
                ldmark_c, rate = self._random_enlarge_ldmark(ldmark_c)
            else:
                rate = 1

            img_p = self.transform(img_p)
            img_c = self.transform(img_c)

            psmap_m2c = self._cal_psmap(self.ldmark_mean, ldmark_c)
            psmap_m2p = self._cal_psmap(self.ldmark_mean, ldmark_p)
            psmap_p2c = self._cal_psmap(ldmark_p, ldmark_c)

            item = {
                'name': os.path.basename(os.path.dirname(img_p_path)),
                'filename': os.path.basename(img_p_path)[:-4],
                'img_p': img_p,
                'img_c': img_c,
                'ldmark_p': ldmark_p,
                'ldmark_c': ldmark_c,
                'psmap_p2c': psmap_p2c,
                'psmap_m2c': psmap_m2c,
                'psmap_m2p': psmap_m2p,
                'enlarge_rate': rate
            }
        else:
            img_p_path = self.img_list[index]
            img_p = self._load_img(img_p_path)
            name = os.path.basename(os.path.dirname(img_p_path))

            img_p = self.transform(img_p)

            item = {
                'img_p': img_p,
                'name': name,
                'filename': os.path.basename(img_p_path)[:-4]
            }

        return item

    def __len__(self):
        return self.size


def make_dataset(args):
    list_total = os.listdir(os.path.join(args.data_root, 'image'))
    train_list = random.sample(list_total, len(list_total) // 2)
    if args.mode == 'train':
        train_set = WCDataSet(args, train_list, 'train')
        return train_set
    else:
        test_list = list(set(list_total).difference(set(train_list)))
        test_set = WCDataSet(args, test_list, 'test')
        return test_set