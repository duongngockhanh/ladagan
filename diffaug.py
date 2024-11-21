"""Differentiable Augmentation for PyTorch.

Reference:
  - [Differentiable Augmentation for Data-Efficient GAN Training](
      https://arxiv.org/abs/2006.10738) (NeurIPS 2020)
"""
import torch
import torch.nn.functional as F


def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
    return x


def rand_brightness(x):
    magnitude = torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5
    x = x + magnitude
    return x


def rand_saturation(x):
    magnitude = torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_contrast(x):
    magnitude = torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_translation(x, ratio=0.125):
    batch_size = x.size(0)
    image_size = x.size()[2:]
    shift = [int(s * ratio + 0.5) for s in image_size]
    translation_x = torch.randint(-shift[0], shift[0] + 1, size=[batch_size, 1], device=x.device)
    translation_y = torch.randint(-shift[1], shift[1] + 1, size=[batch_size, 1], device=x.device)
    
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(batch_size, device=x.device),
        torch.arange(image_size[0], device=x.device),
        torch.arange(image_size[1], device=x.device),
    )
    
    grid_x = grid_x + translation_x.reshape(-1, 1, 1)
    grid_y = grid_y + translation_y.reshape(-1, 1, 1)
    
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    grid_x = grid_x.float() / (image_size[0] - 1) * 2 - 1
    grid_y = grid_y.float() / (image_size[1] - 1) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=3)
    x = F.grid_sample(x_pad, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return x


def rand_cutout(x, ratio=0.5):
    batch_size = x.size(0)
    image_size = x.size()[2:]
    cutout_size = [int(s * ratio + 0.5) for s in image_size]
    offset_x = torch.randint(0, image_size[0] + (1 - cutout_size[0] % 2), size=[batch_size, 1, 1], device=x.device)
    offset_y = torch.randint(0, image_size[1] + (1 - cutout_size[1] % 2), size=[batch_size, 1, 1], device=x.device)
    
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(batch_size, device=x.device),
        torch.arange(cutout_size[0], device=x.device),
        torch.arange(cutout_size[1], device=x.device),
    )
    
    cutout_grid = torch.stack([
        grid_batch,
        grid_x + offset_x - cutout_size[0] // 2,
        grid_y + offset_y - cutout_size[1] // 2,
    ], dim=-1)
    
    mask_shape = (batch_size, image_size[0], image_size[1])
    cutout_grid = torch.clamp(cutout_grid, min=0)
    cutout_grid = torch.clamp(cutout_grid, max=torch.tensor([batch_size-1, image_size[0]-1, image_size[1]-1])[None, None, None, :])
    
    mask = torch.ones(mask_shape, device=x.device)
    for i in range(batch_size):
        mask[i][cutout_grid[i, :, :, 1], cutout_grid[i, :, :, 2]] = 0
    mask = mask.unsqueeze(1)
    x = x * mask
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
