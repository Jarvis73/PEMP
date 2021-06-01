from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D
from sacred import Ingredient

from networks import backbones

net_ingredient = Ingredient("net", save_git_info=False)
pretrained_weights = {
    "vgg16": Path(__file__).parents[1] / "data/vgg16-397923af.pth",
    "resnet50": Path(__file__).parents[1] / "data/resnet50-19c8e357.pth",
    "resnet101": Path(__file__).parents[1] / "data/resnet101-5d3b4d8f.pth"
}
backbone_error = "Not supported backbone '{}'. [vgg16, resnet50, resnet101]"


@net_ingredient.config
def net_config():
    dist_scalar = 20                        # int, a factor multiplied to cosine distance results
    init_channels = 3                       # int, input channels of the model
    out_channels = 512                      # int, output channels of the feature extractor
    backbone = "resnet50"                   # str, structure of the feature extractor. [vgg16, resnet50, resnet101]
    protos = 3                              # int, number of prototypes per class
    drop_rate = 0.1                         # float, drop rate used in the DropBlock of the purifier
    block_size = 4                          # int, block size used in the DropBlock of the purifier


@net_ingredient.config_hook
def net_hook(config, command_name, logger):
    net_cfg = config["net"]
    if net_cfg["backbone"] not in list(pretrained_weights.keys()):
        raise ValueError(backbone_error.format(net_cfg["backbone"]))
    return {}


class PEMPStage1(backbones.BaseModel, nn.Module):
    """
    Stage 1 of the proposed Prior-Enhanced network with Meta-Prototypes.

    Parameters
    ----------
    backbone, init_channels, out_channels, protos, drop_rate, block_size:
        [Config]

    Notes
    -----
    Parameters denoted with '[Config]' are autofilled by sacred configuration and there is
    no need to manually input.
    """
    @net_ingredient.capture
    def __init__(self, logger,
                 backbone, init_channels, out_channels, protos, drop_rate, block_size):
        super(PEMPStage1, self).__init__()
        pretrained = pretrained_weights[backbone]

        if backbone == "vgg16":
            self.encoder = nn.Sequential(OrderedDict([
                ('backbone', backbones.VGG16(init_channels, pretrained, lastRelu=False))
            ]))
            self.__class__.__name__ = "PEMP_Stage1/VGG16"
        elif backbone == "resnet50":
            self.encoder = nn.Sequential(OrderedDict([
                ('backbone', backbones.ResNet(init_channels,
                                              backbones.BottleNeck,
                                              layers=[3, 4, 6],
                                              freeze_bn=True,
                                              ret_features=False,
                                              pretrained=pretrained)),
                ('purifier', nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=True),
                    nn.ReLU(),
                    DropBlock2D(drop_rate, block_size),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(),
                    DropBlock2D(drop_rate, block_size),
                    backbones.ASPPV2(inc=256, midc=256, outc=out_channels, drop_rate=drop_rate, block_size=block_size)))
            ]))
            self.__class__.__name__ = "PEMP_Stage1/Resnet50"
        elif backbone == "resnet101":
            self.encoder = nn.Sequential(OrderedDict([
                ('backbone', backbones.ResNet(init_channels,
                                              backbones.BottleNeck,
                                              layers=[3, 4, 23],
                                              freeze_bn=True,
                                              ret_features=False,
                                              pretrained=pretrained)),
                ('purifier', nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=True),
                    nn.ReLU(),
                    DropBlock2D(drop_rate, block_size),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(),
                    DropBlock2D(drop_rate, block_size),
                    backbones.ASPPV2(inc=256, midc=256, outc=out_channels, drop_rate=drop_rate, block_size=block_size)))
            ]))
            self.__class__.__name__ = "PEMP_stage1/Resnet101"
        else:
            raise ValueError(backbone_error.format(backbone))

        if protos > 0:
            self.ctr = torch.nn.Parameter(torch.rand(out_channels, protos * 2), requires_grad=True)
        else:
            self.ctr = None

        logger.info(f"           ==> Model {self.__class__.__name__} created")

    @net_ingredient.capture
    def forward(self, sup_img, sup_mask, qry_img, out_shape=None, ret_ind=False):
        """
        Parameters
        ----------
        sup_img: torch.Tensor,
            Support images of the shape [B, S, 3, H, W] and dtype float32
        sup_mask: torch.Tensor
            Support labels of the shape [B, S, 2, H, W] and dtype float32, shape `2` means [fg, bg]
        qry_img: torch.Tensor
            Query images of the shape [B, Q, 3, H, W] and dtype float32
        out_shape : tuple
            Output size of the prediction. A tuple of two integers.
        ret_ind: bool
            Return prediction indices (argmax) for generating the response map.

        Returns
        -------
        pred: torch.Tensor
            Prediction of the network of the shape [BQ, 2, h, w] and dtype float32
        response: torch.Tensor
            [optinal] Response map of the query prediction to multiple prototypes.
            The shape is [BQ, h, w] and the dtype is int64

        """
        B, S, channel, H, W = sup_img.size()
        Q = qry_img.size(1)

        img_cat = torch.cat((sup_img, qry_img), dim=1).view(B * (S + Q), channel, H, W)
        features = self.encoder(img_cat)                                                    # [B(S + Q), c, h, w]
        _, c, h, w = features.size()
        features = features.view(B, S + Q, c, h, w)                                         # [B, S + Q, c, h, w]

        sup_fts = features[:, :S]                                                           # [B, S, c, h, w]
        qry_fts = features[:, S:]                                                           # [B, Q, c, h, w]
        sup_mask = sup_mask.view(B * S, 2, H, W)                                            # [BS, 2, H, W]
        sup_mask = F.interpolate(sup_mask, (h, w), mode="nearest")                          # [BS, 2, h, w]
        sup_mask_fg, sup_mask_bg = sup_mask.unbind(dim=1)                                   # [BS, h, w]

        pred = self.mpm(sup_fts, qry_fts, sup_mask_fg, sup_mask_bg, ret_ind)                # [BQ, 2, h, w]

        if out_shape is None:
            out_shape = (H, W)

        if ret_ind:
            pred, response = pred
            output = F.interpolate(pred, out_shape, mode='bilinear', align_corners=True)    # [BQ, 2, H, W]
            response = F.interpolate(response.unsqueeze(dim=1).float(), out_shape, mode='nearest')
            response = response.squeeze(dim=1).long()                                       # [BQ, H, W]
            return output, response
        else:
            output = F.interpolate(pred, out_shape, mode='bilinear', align_corners=True)    # [BQ, 2, H, W]
            return output

    @net_ingredient.capture
    def mpm(self, sup_fts, qry_fts, sup_fg, sup_bg, ret_ind,
            protos):
        """
        Implementation of the Meta-Prototype Module and final prediction.

        Parameters
        ----------
        sup_fts: torch.Tensor
            Support images of the shape [B, S, c, h, w] and dtype float32
        qry_fts: torch.Tensor
            Query images of the shape [B, Q, c, h, w] and dtype float32
        sup_fg: torch.Tensor
            Support foreground mask of the shape [BS, h, w] and dtype float32
        sup_bg: torch.Tensor
            Support background mask of the shape [BS, h, w] and dtype float32
        ret_ind: bool
            Return prediction indices (argmax) for generating the response map.
        protos: int
            [Config]

        Returns
        -------
        pred: torch.Tensor
            Prediction of the network of the shape [BQ, 2, h, w] and dtype float32
        response: torch.Tensor
            [optinal] Response map of the query prediction to multiple prototypes.
            The shape is [BQ, h, w] and the dtype is int64

        """
        B, S, c, h, w = sup_fts.shape
        sup_fts = sup_fts.reshape(-1, c, h * w)
        qry_fts = qry_fts.reshape(-1, c, 1, h, w)
        sup_fg = sup_fg.view(-1, 1, h * w)  # [BS, 1, hw]
        sup_bg = sup_bg.view(-1, 1, h * w)  # [BS, 1, hw]

        if self.ctr is not None:
            ctr = self.ctr.view(1, c, protos * 2)                                                   # [1, c, 2p]
            mask = torch.stack((sup_fg, sup_bg), dim=1)                                             # [BS, 2, 1, hw]

            D = -((sup_fts.unsqueeze(dim=2) - ctr.unsqueeze(dim=3)) ** 2).sum(dim=1)                # [BS, 2p, hw]
            D = D.view(-1, 2, protos, h * w)                                                        # [BS, 2, p, hw]
            D = (torch.softmax(D, dim=2) * mask).view(-1, 1, protos * 2, h * w)                     # [BS, 1, 2p, hw]
            masked_fts = sup_fts.view(-1, c, 1, h * w) * D                                          # [BS, c, 2p, hw]
            ctr = (masked_fts.sum(dim=3) / (D.sum(dim=3) + 1e-6)).view(B, S, c, 2, protos)          # [B, S, c, 2, p]
            ctr = ctr.transpose(3, 4).reshape(B, S, c * protos, 2)                                  # [B, S, cp, 2]
            ctr = ctr.mean(dim=1)                                                                   # [B, cp, 2]

            fg_proto, bg_proto = ctr.view(B, c, protos, 2).unbind(dim=3)                            # [B, c, p]
            max_v = self.compute_similarity(fg_proto, bg_proto, qry_fts).max(dim=2)
            pred = max_v.values                                                                     # [BQ, 2, h, w]

            if ret_ind:
                ind = max_v.indices
                response = ind[:, 0].clone()                 # background
                select = pred.argmax(dim=1) == 1
                response[select] = ind[:, 1][select] + 3     # foreground
                return pred, response
        else:
            fg_vecs = torch.sum(sup_fts * sup_fg, dim=-1) / (sup_fg.sum(dim=-1) + 1e-5)     # [BS, c]
            bg_vecs = torch.sum(sup_fts * sup_bg, dim=-1) / (sup_bg.sum(dim=-1) + 1e-5)     # [BS, c]
            fg_proto = fg_vecs.view(B, S, c).mean(dim=1)
            bg_proto = bg_vecs.view(B, S, c).mean(dim=1)
            pred = self.compute_similarity(fg_proto, bg_proto, qry_fts.view(-1, c, h, w))   # [BQ, 2, h, w]

        return pred

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
        dist_scalar: float, int
            [Config]

        Returns
        -------
        pred: torch.Tensor
            Prediction of the query image segmentation with the shape of [BQ, 2, h, 2]

        """
        fg_distance = F.cosine_similarity(
            qry_fts, fg_proto[..., None, None], dim=1) * dist_scalar        # [BQ, 3, h, w]
        bg_distance = F.cosine_similarity(
            qry_fts, bg_proto[..., None, None], dim=1) * dist_scalar        # [BQ, 3, h, w]
        pred = torch.stack((bg_distance, fg_distance), dim=1)               # [BQ, 2, 3, h, w]
        return pred


ModelClass = PEMPStage1

