import os
import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms.functional as transF
from PIL import Image
import torchvision.transforms as transforms

from utils import make_init_field, load_filenames, warp_position_map

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

flip_index = [0, 3, 2, 1, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 16, 15, 14]


def make_dataset(args):
    list_total = os.listdir(os.path.join(args.data_root, 'image'))
    train_list = random.sample(list_total, len(list_total) // 2)
    if args.mode == 'train':
        return WCDataSet(args, train_list, 'train')
    else:
        test_list = list(set(list_total).difference(set(train_list)))
        return WCDataSet(args, test_list, 'test')


def load_img(path):
    return Image.open(path).convert('RGB')


def load_landmark(path):
    return torch.from_numpy(np.loadtxt(path, delimiter='\t')).float()


def cal_field(src, dst):
    field_y, field_x = warp_position_map(src, dst)
    field = np.concatenate((field_y, field_x), axis=2)
    field = torch.from_numpy(field).float()
    return field


def random_horizonal_flip(img, landmark):
    if random.random() < 0.5:
        img_hflip = transF.hflip(img)
        landmark[:, 0] = 256 - landmark[:, 0]
        landmark_hflip = torch.zeros_like(landmark)
        for i in range(landmark.shape[0]):
            landmark_hflip[i] = landmark[flip_index[i]]
        return img_hflip, landmark_hflip
    else:
        return img, landmark


def random_enlarge_landmark(landmark_mean, landmark, p=0.5):
    if random.random() < p:
        rate = random.random() / 5 + 1
    else:
        rate = 1
    landmark = (landmark - landmark_mean) * (rate - 1) + landmark
    return landmark, rate


def random_resize_crop(img, landmark, resize=288):
    if random.random() < 0.5:
        w, h = img.size
        time = resize / w
        img = img.resize((resize, resize), Image.BILINEAR)
        x = random.random() * (time - 1) * w
        y = random.random() * (time - 1) * h
        img_crop = img.crop((x, y, x + w, y + h))
        landmark_crop = landmark * time
        landmark_crop[:, 0] = landmark_crop[:, 0] - x
        landmark_crop[:, 1] = landmark_crop[:, 1] - y
        return img_crop, landmark_crop
    else:
        return img, landmark


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
        self.const_map = make_init_field().squeeze(0)

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
            self.landmark_mean = torch.zeros(17, 2)
            for img_path in self.img_list:
                landmark_path = img_path.replace('image', 'landmark').replace('.jpg', '.txt')
                landmark = load_landmark(landmark_path)
                self.landmark_mean += landmark
            self.landmark_mean /= self.num_p
        else:
            self.size = self.num_p

    def sample_pair(self, same_id=True):
        name1 = random.choice(self.name_list)
        img_p_path = random.choice(self.p_dir[name1])
        landmark_p_path = img_p_path.replace('image', 'landmark').replace('.jpg', '.txt')

        if same_id:
            img_c_path = random.choice(self.c_dir[name1])
        else:
            name2 = random.choice(self.name_list)
            img_c_path = random.choice(self.c_dir[name2])
        landmark_c_path = img_c_path.replace('image', 'landmark').replace('.jpg', '.txt')
        return img_p_path, img_c_path, landmark_p_path, landmark_c_path

    def __getitem__(self, index):
        if self.mode == 'train':
            img_p_path, img_c_path, landmark_p_path, landmark_c_path = self.sample_pair(same_id=self.same_id)

            img_p = load_img(img_p_path)
            img_c = load_img(img_c_path)
            landmark_p = load_landmark(landmark_p_path)
            landmark_c = load_landmark(landmark_c_path)

            if self.hflip:
                img_p, landmark_p = random_horizonal_flip(img_p, landmark_p)
                img_c, landmark_c = random_horizonal_flip(img_c, landmark_c)

            if self.resize_crop:
                img_p, landmark_p = random_resize_crop(img_p, landmark_p)
                img_c, landmark_c = random_resize_crop(img_c, landmark_c)

            if self.enlarge:
                landmark_c, _ = random_enlarge_landmark(self.landmark_mean, landmark_c)

            img_p = self.transform(img_p)
            img_c = self.transform(img_c)

            field_m2c = cal_field(self.landmark_mean, landmark_c)
            field_m2p = cal_field(self.landmark_mean, landmark_p)
            field_p2c = cal_field(landmark_p, landmark_c)

            item = {
                'name': os.path.basename(os.path.dirname(img_p_path)),
                'filename': os.path.basename(img_p_path)[:-4],
                'img_p': img_p,
                'img_c': img_c,
                'field_p2c': field_p2c,
                'field_m2c': field_m2c,
                'field_m2p': field_m2p,
            }
        else:
            img_p_path = self.img_list[index]
            img_p = load_img(img_p_path)
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

