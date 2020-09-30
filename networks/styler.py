import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.modules import Conv2dBlock, ResBlocks, LinearBlock, AdaResBlocks


class ContentEncoder(nn.Module):
    def __init__(self, input_dim=3, dim=64, num_res=3, downs=2, norm='in', activation='relu', pad_type='reflect'):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activation, pad_type=pad_type)]
        for i in range(downs):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activation, pad_type=pad_type)]
            dim *= 2
        self.model += [ResBlocks(num_res, dim, norm=norm, activation='relu', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class StyleEncoder(nn.Module):
    def __init__(self, input_dim=3, dim=64, downs=2, style_dim=8, norm='none', activation='relu', pad_type='reflect'):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activation, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activation, pad_type=pad_type)]
            dim *= 2
        for i in range(downs - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activation, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)]
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.norm = nn.BatchNorm2d(style_dim)

    def forward(self, x):
        x = self.model(x)
        # if norm:
        #     x = self.norm(x)
        return x


class StyleController(nn.Module):
    def __init__(self, style_dim=8, dim=256, norm='ln', activation='relu'):
        super(StyleController, self).__init__()
        self.model = []
        self.model += [LinearBlock(style_dim, dim, norm=norm, activation=activation)]
        self.model += [LinearBlock(dim, dim, norm=norm, activation=activation)]
        self.model = nn.Sequential(*self.model)
        self.fc_gamma = LinearBlock(256, 256, norm='none', activation='none')
        self.fc_beta = LinearBlock(256, 256, norm='none', activation='none')

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        gamma = self.fc_gamma(x).unsqueeze(2).unsqueeze(3)
        beta = self.fc_beta(x).unsqueeze(2).unsqueeze(3)
        return gamma, beta


class Decoder(nn.Module):
    def __init__(self, output_dim=3, dim=256, num_res=3, ups=2, restype='adalin', norm='lin', activation='relu',
                 pad_type='reflect'):
        super(Decoder, self).__init__()
        self.res = AdaResBlocks(num_res, dim, restype=restype)
        self.model = []
        for i in range(ups):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=norm, activation=activation, pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x, gamma, beta):
        x = self.res(x, gamma, beta)
        return self.model(x)


class Gen_Style(nn.Module):
    def __init__(self, args):
        super(Gen_Style, self).__init__()
        input_dim = 3
        dim = 64
        downs = 2
        num_res = 4
        restype = args.restype
        style_dim = args.style_dim
        # self.norm = args.norm_style
        norm_d = 'lin' if restype == 'adalin' else 'ln'
        self.encoder_c = ContentEncoder(input_dim, dim, num_res, downs, norm='in', activation='relu',
                                        pad_type='reflect')
        self.encoder_s = StyleEncoder(input_dim, dim, args.down_es, style_dim, norm='none', activation='relu',
                                      pad_type='reflect')
        latent_dim = self.encoder_c.output_dim
        self.decoder = Decoder(input_dim, latent_dim, num_res, downs, restype=restype, norm=norm_d, activation='relu',
                               pad_type='reflect')
        self.style_controller = StyleController(style_dim, latent_dim, norm='ln', activation='relu')

    def encode(self, x):
        content = self.encoder_c(x)
        style = self.encoder_s(x)
        return content, style

    def decode(self, content, style):
        gamma, beta = self.style_controller(style)
        output = self.decoder(content, gamma, beta)
        return output

    def forward(self, img_p, s):
        content, _ = self.encode(img_p)
        output = self.decode(content, s)
        return output


class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()
        dim = 64
        n_layers = 5
        model = []
        model += [Conv2dBlock(3, 64, 4, 2, 1, norm='sn', activation='lrelu', use_bias=True, pad_type='reflect')]

        for i in range(1, n_layers - 2):
            model += [
                Conv2dBlock(dim, dim * 2, 4, 2, 1, norm='sn', activation='lrelu', use_bias=True, pad_type='reflect')]
            dim = dim * 2

        model += [Conv2dBlock(dim, dim * 2, 4, 1, 1, norm='sn', activation='lrelu', use_bias=True, pad_type='reflect')]

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(dim * 2, 1, kernel_size=4, stride=1, padding=1, bias=False))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        x = self.pad(x)
        x = self.conv(x)
        return x

    def calc_dis_loss(self, img_real, img_fake):
        # calculate the loss to train D
        logit_real = self.forward(img_real)
        logit_fake = self.forward(img_fake)
        loss = F.mse_loss(logit_real, torch.ones_like(logit_real).cuda()) \
               + F.mse_loss(logit_fake, torch.zeros_like(logit_fake).cuda())
        return loss

    def calc_gen_loss(self, img_fake):
        # calculate the loss to train G
        logit_fake = self.forward(img_fake)
        loss = F.mse_loss(logit_fake, torch.ones_like(logit_fake).cuda())
        return loss


class Styler(nn.Module):
    def __init__(self, args):
        super(Styler, self).__init__()
        self.gen = Gen_Style(args)
        self.dis = Dis()

    def encode(self, x):
        return self.gen.encode(x)

    def decode(self, content, s):
        return self.gen.deocde(content, s)

    def forward(self, img_p, s):
        output = self.gen(img_p, s)
        return output

    def save(self, dir, step):
        gen_name = os.path.join(dir, 'gen_%08d.pt' % (step + 1))
        torch.save(self.gen.state_dict(), gen_name)
        dis_name = os.path.join(dir, 'dis_%08d.pt' % (step + 1))
        torch.save(self.dis.state_dict(), dis_name)

    def load(self, path):
        state_dict = torch.load(path)
        self.gen.load_state_dict(state_dict)

