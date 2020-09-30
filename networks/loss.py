import torch


def l1_loss(input, target):
    return torch.mean(torch.abs(input - target))


def mse_loss(input, target):
    return torch.mean((input-target)**2)


def kl_loss(logvar, mu):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)


def tv_loss(img):
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = h_variance + w_variance
    return loss
