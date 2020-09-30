import os

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import make_constant_map
from .modules import Conv2dBlock, LinearBlock


class Encoder(nn.Module):
    def __init__(self, input_dim=3, dim=64, latent_dim=32, norm='bn', activation='relu', pad_type='reflect'):
        super(Encoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activation, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activation, pad_type=pad_type)]
            dim *= 2
        count = int(math.log2(dim / latent_dim))
        for i in range(count):
            self.model += [Conv2dBlock(dim, dim // 2, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type)]
            dim //= 2
        self.model = nn.Sequential(*self.model)

        self.pool = []
        self.pool += [
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(latent_dim)
        ]
        self.pool = nn.Sequential(*self.pool)

    def forward(self, x, norm=False):
        x = self.model(x)
        embedding = self.pool(x)
        return x, embedding


class Decoder(nn.Module):
    def __init__(self, dim=32, output_dim=3, norm='bn', activation='relu', pad_type='reflect'):
        super(Decoder, self).__init__()
        self.model = []
        count = 8 - int(math.log2(dim))
        for i in range(count):
            self.model += [Conv2dBlock(dim, dim * 2, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type)]
            dim *= 2
        for i in range(2):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=norm, activation=activation, pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class WarpEncoder(nn.Module):
    def __init__(self, input_dim=2, dim=64, downs=4, code_dim=8, last_bn=True, norm='bn', activation='relu',
                 pad_type='reflect'):
        super(WarpEncoder, self).__init__()

        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activation, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activation, pad_type=pad_type)]
            dim *= 2
        for i in range(downs - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activation, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(dim, code_dim, 1, 1, 0)]
        if last_bn:
            self.model += [nn.BatchNorm2d(code_dim)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class WarpDecoder(nn.Module):
    def __init__(self, latent_dim=96, output_dim=2, output_size=256, dim=256, ups=4, norm='bn', activation='relu',
                 pad_type='reflect'):
        super(WarpDecoder, self).__init__()
        self.init_size = output_size // (2 ** ups)
        self.init_dim = dim
        self.linear = []
        self.linear += [LinearBlock(latent_dim, dim * self.init_size ** 2, norm=norm, activation=activation)]
        self.linear = nn.Sequential(*self.linear)

        self.conv = []

        for i in range(ups):
            self.conv += [nn.Upsample(scale_factor=2),
                          Conv2dBlock(dim, dim // 2, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type),
                          Conv2dBlock(dim // 2, dim // 2, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)
                          ]
            dim //= 2

        self.conv += [Conv2dBlock(dim, output_dim, 5, 1, 2, norm='none', activation='none', pad_type=pad_type)]
        self.conv = nn.Sequential(*self.conv)

    def forward(self, z):
        z = z.view(z.size(0), -1)
        out = self.linear(z)
        out = out.view(out.shape[0], self.init_dim, self.init_size, self.init_size)
        out = self.conv(out)
        return out


class Warper(nn.Module):
    def __init__(self, args):
        super(Warper, self).__init__()
        self.enc = Encoder(latent_dim=args.embedding_dim)
        self.dec = Decoder(dim=args.embedding_dim)
        self.encoder_w = WarpEncoder(input_dim=2, dim=64, downs=args.ups_dw, code_dim=args.warp_dim,
                                     last_bn=args.last_bn)
        self.decoder_w = WarpDecoder(latent_dim=(args.embedding_dim + args.warp_dim), ups=args.ups_dw,
                                     output_size=args.psmap)
        self.const_map = make_constant_map(256).cuda()
        self.factor = 256 // args.psmap

    def encode(self, psmap):
        psmap = F.interpolate(psmap.permute(0, 3, 1, 2), scale_factor=1 / self.factor, mode='bilinear',
                              align_corners=True)
        z = self.encoder_w(psmap)
        return z

    def decode(self, embedding, z, scale=1.0):
        flow = torch.cat((embedding, z), dim=1)
        flow = self.decoder_w(flow)
        flow = F.interpolate(flow, scale_factor=self.factor, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        out = self.const_map + scale * flow
        return flow, out

    def forward(self, img_p, z, scale=1.0):
        _, embedding = self.enc(img_p)
        flow, psmap_pred = self.decode(embedding, z, scale)
        output = F.grid_sample(img_p, psmap_pred, align_corners=True)
        return output, psmap_pred, flow

    def save(self, dir, step):
        warper_name = os.path.join(dir, 'warper_%08d.pt' % (step + 1))
        torch.save(self.state_dict(), warper_name)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
