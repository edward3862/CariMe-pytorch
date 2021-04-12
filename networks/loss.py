import torch


def l1_loss(input, target):
    return torch.mean(torch.abs(input - target))


def mse_loss(input, target):
    return torch.mean((input-target)**2)


def tv_loss(img):
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = h_variance + w_variance
    return loss
