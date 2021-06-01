from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import backbones
from networks.pemp_stage1 import net_ingredient, PEMPStage1, pretrained_weights, backbone_error

PriorNet = PEMPStage1


@net_ingredient.config
def priornet_config():
    backbone2 = "resnet50"                  # str, structure of the feature extractor. Default to stage1's backbone. [vgg16, resnet50, resnet101]
    protos2 = 3                             # int, number of prototypes per class
    drop_rate2 = 0.5                        # float, drop rate used in the Dropout of the purifier
    cm = True                               # bool, use communication module


class PEMPStage2(backbones.BaseModel, nn.Module):
    """
    Stage 1 of the proposed Prior-Enhanced network with Meta-Prototypes.

    Parameters
    ----------
    shot: int
        Number of support images in each episode
    query: int
        Number of query images in each episode. Fixed to 1.
    backbone, backbone2, init_channels, out_channels, protos2, drop_rate2, cm:
        [Config]

    Notes
    -----
    Parameters denoted with '[Config]' are autofilled by sacred configuration and there is
    no need to manually input.
    """
    @net_ingredient.capture
    def __init__(self, shot, query, logger,
                 backbone, backbone2, init_channels, out_channels, protos2, drop_rate2, cm):
        super(PEMPStage2, self).__init__()
        if not backbone2:
            backbone2 = backbone
        pretrained = pretrained_weights[backbone2]

        if backbone2 == "vgg16":
            self.encoder = nn.Sequential(OrderedDict([
                ('backbone', backbones.VGG16CM(init_channels + 1,
                                         pretrained=pretrained,
                                         lastRelu=False,
                                         shot_query=shot + query))
            ]))
            self.__class__.__name__ = "PEMP_Stage2/VGG16" + cm * "+CM"
        elif backbone2 == "resnet50":
            self.encoder = nn.Sequential(OrderedDict([
                ('backbone', backbones.ResNetCM(init_channels + 1,
                                                backbones.BottleNeck,
                                                layers=[3, 4, 6],
                                                freeze_bn=True,
                                                ret_features=False,
                                                pretrained=pretrained,
                                                shot_query=shot + query)),
                ('purifier', nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Dropout2d(drop_rate2),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(),
                    nn.Dropout2d(drop_rate2),
                    backbones.ASPP(inc=256, midc=256, outc=out_channels, drop_rate=drop_rate2)))
            ]))
            self.__class__.__name__ = f"PEMP_Stage2/Resnet50" + cm * "+CM"
        elif backbone2 == "resnet101":
            self.encoder = nn.Sequential(OrderedDict([
                ('backbone', backbones.ResNetCM(init_channels + 1,
                                                backbones.BottleNeck,
                                                layers=[3, 4, 23],
                                                freeze_bn=True,
                                                ret_features=False,
                                                pretrained=pretrained,
                                                shot_query=shot + query)),
                ('purifier', nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Dropout2d(drop_rate2),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(),
                    nn.Dropout2d(drop_rate2),
                    backbones.ASPP(inc=256, midc=256, outc=out_channels, drop_rate=drop_rate2)))
            ]))
            self.__class__.__name__ = f"PEMP_Stage2/Resnet50" + cm * "+CM"
        else:
            raise ValueError(backbone_error.format(backbone2))

        if protos2 > 0:
            self.ctr = torch.nn.Parameter(torch.rand(out_channels, protos2 * 2), requires_grad=True)
        else:
            self.ctr = None

        logger.info(f"           ==> Model {self.__class__.__name__} created")

    def forward(self, sup_img, sup_mask, qry_img, qry_prior, out_shape=None, ret_ind=False):
        """
        Parameters
        ----------
        sup_img: torch.Tensor,
            Support images of the shape [B, S, 3, H, W] and dtype float32
        sup_mask: torch.Tensor
            Support labels of the shape [B, S, 2, H, W] and dtype float32
        qry_img: torch.Tensor
            Query images of the shape [B, Q, 3, H, W] and dtype float32
        qry_prior: torch.Tensor
            Query prior of the shape [BQ, 1, H, W] width dtype float32
        out_shape : tuple
            Output size of the prediction. A tuple of two integers.
        ret_ind: bool
            Return indices of prediction map for visualization.

        Notes
        -----
        Boundary pixels (mask=255) are ignored during training/testing. Therefore sup_mask_fg and
        sup_mask_bg are not complementary and both of them need to be provided.

        """
        B, S, channel, H, W = sup_img.size()
        Q = qry_img.size(1)

        # Input channel 1-3: Images
        img_cat = torch.cat((sup_img, qry_img), dim=1).view(B * (S + Q), channel, H, W)
        # Input channel 4: Pirors
        sup_prior = sup_mask[:, :, :1]   # support masks used as the prior              # [B, S, 1, H, W]
        qry_prior = qry_prior.view(B, Q, *qry_prior.shape[-3:])                         # [B, Q, 1, H, W]
        prior_cat = torch.cat((sup_prior, qry_prior.float()), dim=1)
        prior_cat = prior_cat.view(B * (S + Q), 1, H, W)

        inputs = torch.cat((img_cat, prior_cat), dim=1)                                 # [B(S + Q), 4, H, W]
        features = self.encoder((inputs, prior_cat))                                    # [B(S + Q), c, h, w]
        _, c, h, w = features.size()
        features = features.view(B, S + Q, c, h, w)                                     # [B, S + Q, c, h, w]

        sup_fts = features[:, :S]                                                       # [B, S, c, h, w]
        qry_fts = features[:, S:]                                                       # [B, Q, c, h, w]
        sup_mask = sup_mask.view(B * S, 2, H, W)                                        # [BS, 2, H, W]
        sup_mask = F.interpolate(sup_mask, (h, w), mode="nearest")                      # [BS, 2, h, w]
        sup_mask_fg, sup_mask_bg = sup_mask.unbind(dim=1)                               # [BS, h, w]

        pred = self.mpm(sup_fts, qry_fts, sup_mask_fg, sup_mask_bg, ret_ind)            # [BQ, 2, h, w]

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
            protos2):
        B, S, c, h, w = sup_fts.shape
        sup_fts = sup_fts.reshape(-1, c, h * w)
        qry_fts = qry_fts.reshape(-1, c, 1, h, w)
        sup_fg = sup_fg.view(-1, 1, h * w)                                                      # [BS, 1, hw]
        sup_bg = sup_bg.view(-1, 1, h * w)                                                      # [BS, 1, hw]

        if self.ctr is not None:
            ctr = self.ctr.view(1, c, protos2 * 2)                                              # [1, c, 2p]
            mask = torch.stack((sup_fg, sup_bg), dim=1)                                         # [BS, 2, 1, hw]
    
            D = -((sup_fts.unsqueeze(dim=2) - ctr.unsqueeze(dim=3)) ** 2).sum(dim=1)            # [BS, 2p, hw]
            D = D.view(-1, 2, protos2, h * w)                                                   # [BS, 2, p, hw]
            D = (torch.softmax(D, dim=2) * mask).view(-1, 1, protos2 * 2, h * w)                # [BS, 1, 2p, hw]
            masked_fts = sup_fts.view(-1, c, 1, h * w) * D                                      # [BS, c, 2p, hw]
            ctr = (masked_fts.sum(dim=3) / (D.sum(dim=3) + 1e-6)).view(B, S, c, 2, protos2)     # [B, S, c, 2, p]
            ctr = ctr.transpose(3, 4).reshape(B, S, c * protos2, 2)                             # [B, S, cp, 2]
            ctr = ctr.mean(dim=1)                                                               # [B, cp, 2]

            self.adaptive_p = ctr.view(B, c, protos2, 2).transpose(2, 3).reshape(B, c, -1)      # [B, c, 2p]
            fg_proto, bg_proto = ctr.view(B, c, protos2, 2).unbind(dim=3)                       # [B, c, p]
            max_v = self.compute_similarity(fg_proto, bg_proto, qry_fts).max(dim=2)
            pred = max_v.values                                                                 # [BQ, 2, h, w]

            if ret_ind:
                ind = max_v.indices                                                             # [BQ, 2, h, w]
                response = ind[:, 0].clone()                 # background
                select = pred.argmax(dim=1) == 1
                response[select] = ind[:, 1][select] + 3     # foreground
                return pred, response
        else:
            fg_vecs = torch.sum(sup_fts * sup_fg, dim=-1) / (sup_fg.sum(dim=-1) + 1e-5)         # [BS, c]
            bg_vecs = torch.sum(sup_fts * sup_bg, dim=-1) / (sup_bg.sum(dim=-1) + 1e-5)         # [BS, c]
            fg_proto = fg_vecs.view(B, S, c).mean(dim=1)
            bg_proto = bg_vecs.view(B, S, c).mean(dim=1)
            pred = self.compute_similarity(fg_proto, bg_proto, qry_fts.view(-1, c, h, w))       # [BQ, 2, h, w]
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
            [Config] Distance scalar when computing similarity.

        Returns
        -------
        pred: torch.Tensor
            Prediction of the query image segmentation with the shape of [BQ, 2, p, h, w]

        """
        fg_distance = F.cosine_similarity(
            qry_fts, fg_proto[..., None, None], dim=1) * dist_scalar    # [BQ, 3, h, w]
        bg_distance = F.cosine_similarity(
            qry_fts, bg_proto[..., None, None], dim=1) * dist_scalar    # [BQ, 3, h, w]
        pred = torch.stack((bg_distance, fg_distance), dim=1)           # [BQ, 2, 3, h, w]
        return pred


ModelClass = PEMPStage2
