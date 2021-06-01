import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt


def get(cfg):
    if cfg.loss == "ce":
        return nn.CrossEntropyLoss(ignore_index=255)
    elif cfg.loss == "cedt":
        return CELossDT(cfg.sigma)
    else:
        raise ValueError(f"Unsupported loss type, got {cfg.loss}. Please choose from [ce, cedt]")


class CELossDT(object):
    def __init__(self, sigma):
        self.sigma = sigma
        self.loss_obj = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.kernel = torch.ones(1, 1, 3, 3, dtype=torch.float).cuda()

    def boundary2weight(self, boundary):
        bool_boundary = np.around(boundary.detach().cpu().numpy()).astype(np.bool)
        edts = []
        for bdn in bool_boundary:
            edt = distance_transform_edt(np.bitwise_not(bdn))
            edts.append(edt)
        edts_t = torch.from_numpy(np.stack(edts, axis=0))
        weight = torch.exp(-edts_t / self.sigma ** 2) + 1
        return weight.to(dtype=torch.float32, device=boundary.device)

    def __call__(self, inputs, target):
        loss = self.loss_obj(inputs, target)
        mask = torch.zeros_like(target, dtype=torch.float32)
        mask[target == 1] = 1
        mask.unsqueeze_(dim=1)      # [bs, 1, H, W]
        dilated = torch.clamp(F.conv2d(mask, self.kernel, padding=1), 0, 1) - mask
        erosion = mask - torch.clamp(F.conv2d(mask, self.kernel, padding=1) - 8, 0, 1)
        boundary = (dilated + erosion).squeeze(dim=1)     # [bs, H, W]
        weight = self.boundary2weight(boundary)
        weighted_loss = loss * weight
        return weighted_loss.sum() / weight.sum()
