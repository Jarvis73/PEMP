from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sacred import Ingredient

from networks import backbones

net_ingredient = Ingredient("net", save_git_info=False)
pretrained_weights = {
    "vgg16": Path(__file__).parent.parent / "data/vgg16-397923af.pth",
    "resnet50": Path(__file__).parent.parent / "data/resnet50-19c8e357.pth",
}
backbone_error = "Not supported backbone '{}'. [vgg16, resnet50]"


@net_ingredient.config
def net_config():
    """ Network configuration """
    dist_scalar = 20                                # a factor multiplied to cosine distance results
    init_channels = 3                               # input channels of the model
    backbone = "vgg16"                              # model backbone
    out_channels = 512                              # output features


class Baseline(backbones.BaseModel, nn.Module):
    """
    Baseline model of [this research].

    Parameters
    ----------
    backbone, init_channels, out_channels:
        [Config]

    Notes
    -----
    Parameters denoted with '[Config]' are autofilled by sacred configuration
    and there is no need to manually input.
    """
    @net_ingredient.capture
    def __init__(self, logger,
                 backbone, init_channels, out_channels):
        super(Baseline, self).__init__()
        pretrained = pretrained_weights[backbone]

        if backbone == "vgg16":
            self.encoder = nn.Sequential(OrderedDict([
                ('backbone', backbones.VGG16(init_channels, pretrained, lastRelu=False))
            ]))
            self.__class__.__name__ = "Baseline/VGG16"
        elif backbone == "resnet50":
            self.encoder = nn.Sequential(OrderedDict([
                ('backbone', backbones.ResNet(init_channels,
                                              backbones.BottleNeck,
                                              layers=[3, 4, 6],
                                              freeze_bn=True,
                                              ret_features=False,
                                              pretrained=pretrained)),
                ('projection', nn.Conv2d(1024, out_channels, kernel_size=1, stride=1, bias=True))
            ]))
            self.__class__.__name__ = "Baseline/Resnet50"
        else:
            raise ValueError(backbone_error.format(backbone))

        logger.info(f"           ==> Model {self.__class__.__name__} created")

    def forward(self, sup_img, sup_mask, qry_img, out_shape=None):
        """
        Parameters
        ----------
        sup_img: torch.Tensor,
            Support images of the shape [B, S, 3, H, W] and dtype float32
        sup_mask: torch.Tensor
            Support labels of the shape [B, S, 2, H, W] and dtype float32
        qry_img: torch.Tensor
            Query images of the shape [B, Q, 3, H, W] and dtype float32
        out_shape : tuple
            Output size of the prediction. A tuple of two integers.

        Notes
        -----
        Boundary pixels (mask=255) are ignored during training/testing. Therefore sup_mask_fg and
        sup_mask_bg are not complementary and both of them need to be provided.

        """
        B, S, C, H, W = sup_img.size()
        Q = qry_img.size(1)

        # Extract features
        img_cat = torch.cat((sup_img, qry_img), dim=1).view(B * (S + Q), C, H, W)
        features = self.encoder(img_cat)                                                    # [B(S + Q), c, h, w]
        _, c, h, w = features.size()
        features = features.view(B, S + Q, c, h, w)                                         # [B, S + Q, c, h, w]

        # Align feature size with that of the support mask
        sup_fts = features[:, :S].reshape(B * S, c, h, w)                                   # [BS, c, h, w]
        qry_fts = features[:, S:].reshape(B * Q, c, h, w)                                   # [BQ, c, h, w]
        sup_fts = F.interpolate(sup_fts, (H, W), mode="bilinear", align_corners=True)       # [B, c, H, W]
        sup_mask = sup_mask.view(B * S, 2, H, W)                                            # [BS, 2, H, W]
        sup_mask_fg, sup_mask_bg = sup_mask.split(1, dim=1)                                 # [BS, 1, H, W]

        # Compute prototypes
        fg_vecs = torch.sum(sup_fts * sup_mask_fg, dim=(2, 3)) \
            / (sup_mask_fg.sum(dim=(2, 3)) + 1e-5)                                          # [BS, c]
        bg_vecs = torch.sum(sup_fts * sup_mask_bg, dim=(2, 3)) \
            / (sup_mask_bg.sum(dim=(2, 3)) + 1e-5)                                          # [BS, c]
        fg_proto = fg_vecs.view(B, S, -1).mean(dim=1)                                       # [B, c]
        bg_proto = bg_vecs.view(B, S, -1).mean(dim=1)                                       # [B, c]

        pred = self.compute_similarity(fg_proto, bg_proto, qry_fts)                         # [BQ, 2, h, w]

        if out_shape is None:
            out_shape = (H, W)

        output = F.interpolate(pred, out_shape, mode='bilinear', align_corners=True)        # [BQ, 2, H, W]
        return output

    @net_ingredient.capture
    def compute_similarity(self, fg_proto, bg_proto, qry_fts,
                           dist_scalar):
        """
        Make prediction on the query image according to the foreground prototype
        and the background prototype.

        Parameters
        ----------
        fg_proto: torch.Tensor
            Foreground prototype with the shape of [B, c]
        bg_proto: torch.Tensor
            Background prototype with the shape of [B, c]
        qry_fts: torch.Tensor
            Query feature maps extracted from the backbone with the shape of [BQ, c, h, w]
        dist_scalar: float
            [Config]

        Returns
        -------
        pred: torch.Tensor
            Prediction of the query image segmentation with the shape of [BQ, 2, h, w]

        """
        fg_distance = F.cosine_similarity(
            qry_fts, fg_proto[..., None, None], dim=1) * dist_scalar        # [BQ, h, w]
        bg_distance = F.cosine_similarity(
            qry_fts, bg_proto[..., None, None], dim=1) * dist_scalar        # [BQ, h, w]
        pred = torch.stack((bg_distance, fg_distance), dim=1)               # [BQ, 2, h, w]
        return pred


ModelClass = Baseline
