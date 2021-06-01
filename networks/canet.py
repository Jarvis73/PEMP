from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sacred import Ingredient

from networks import backbones

net_ingredient = Ingredient("net", save_git_info=False)
pretrained_weights = {
    "resnet50": Path(__file__).parent.parent / "data/resnet50-19c8e357.pth"
}


@net_ingredient.config
def net_config():
    """ ==> Network Arguments """
    init_channels = 3                               # input channels of the model
    drop_rate = 0.5
    history = True                                  # bool, use history_mask or not
    freeze_backbone = True                          # bool, freeze backbone parameters or not


class CaNet(backbones.BaseModel, nn.Module):
    """
    CaNet implementation.

    Parameters
    ----------
    init_channels, drop_rate, history, freeze_backbone:
        [Config]

    References
    ----------
    Github:
        https://github.com/icoz69/CaNet
    Paper:
        CANet: Class-Agnostic Segmentation Networks with Iterative Refinement and Attentive
        Few-Shot Learning

    Notes
    -----
    Parameters denoted with '[Config]' are autofilled by sacred configuration and there is
    no need to manually input.
    """
    num_classes = 2

    @net_ingredient.capture
    def __init__(self, logger,
                 init_channels, drop_rate, history, freeze_backbone):
        super(CaNet, self).__init__()
        self.use_history = history
        self.encoder = backbones.ResNet(init_channels,
                                        backbones.BottleNeck,
                                        layers=[3, 4, 6],
                                        freeze_bn=True)

        num_channels = 512 + 1024
        self.layer5 = nn.Sequential(
            nn.Conv2d(num_channels, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate)
        )
        self.layer55 = nn.Sequential(
            nn.Conv2d(256 * 2, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate)
        )

        self.aspp_0 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate)
        )
        self.aspp_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate)
        )
        self.aspp_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=6, dilation=6, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate)
        )
        self.aspp_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate)
        )
        self.aspp_4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate)
        )

        res_input_c = 256 + 2 if self.use_history else 256
        self.residual_1 = nn.Sequential(
            nn.ReLU(),  # Warning: Don't use ``inplace=True`` here due to the residual structure.
            nn.Conv2d(res_input_c, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.residual_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.residual_3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.layer7 = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, bias=True)

        self.init_weights(pretrained_weights["resnet50"])

        logger.info(f"           ==> Model {self.__class__.__name__} created")

    def forward(self, sup_img, sup_mask, qry_img, out_shape=None, history_mask=None):
        """
        Parameters
        ----------
        sup_img: torch.Tensor
            Support images of the shape [B, S, 3, H, W] and dtype float32
        sup_mask: torch.Tensor
            Support labels of the shape [B, S, 2, H, W] and dtype float32
        qry_img: torch.Tensor
            Query images of the shape [B, Q, 3, H, W] and dtype float32
        out_shape : tuple
            Output size of the prediction. A tuple of two integers.
        history_mask: torch.Tensor or None
            Query history predictions of the shape [B, Q, 2, H, W] and dtype float32

        Notes
        -----
        Boundary pixels (mask=255) are ignored during training/testing. Therefore sup_mask_fg and
        sup_mask_bg are not complementary and both of them need to be provided.

        """
        B, S, channel, H, W = sup_img.size()
        Q = qry_img.size(1)

        img_cat = torch.cat((sup_img, qry_img), dim=1).view(B * (S + Q), channel, H, W)
        features = self.encoder(img_cat)

        output = self.relation(features, sup_mask, history_mask)

        if out_shape is not False:
            if out_shape is None:
                out_shape = (H, W)
            output = F.interpolate(output, out_shape, mode='bilinear', align_corners=True)  # [BQ, 2, H, W]

        return output

    def relation(self, features, sup_mask, history_mask):
        f1, f2, f3 = features
        B, S, _, H, W = sup_mask.size()
        Q = history_mask.size(1)

        feat_cat = torch.cat((f2, f3), dim=1)                                           # [B(S + Q), c2 + c3, h, w]
        features = self.layer5(feat_cat)                                                # [B(S + Q), c, h, w]
        _, c, h, w = features.size()
        features = features.view(B, S + Q, c, h, w)                                     # [B, S + Q, c, h, w]
        sup_fts = features[:, :S].reshape((B * S, c, h, w))                             # [BS, c, h, w]
        qry_fts = features[:, S:].reshape((B * Q, c, h, w))                             # [BQ, c, h, w]

        sup_mask = sup_mask[:, :, 0].view(B * S, 1, H, W)                               # [BS, 1, H, W]
        sup_mask = F.interpolate(sup_mask, (h, w), mode='nearest')                      # [BS, 1, h, w]
        z = (sup_fts * sup_mask).sum(dim=(2, 3)) / (sup_mask.sum(dim=(2, 3)) + 1e-5)    # [BS, c]
        z = z.view(B, S, c).mean(dim=1)                                                 # [B, c]
        z = z.view(B, 1, c, 1, 1).repeat(1, Q, 1, h, w).view(B * Q, c, h, w)            # [BQ, c, h, w]
        out = torch.cat((qry_fts, z), dim=1)                                            # [BQ, 2c, h, w]
        out = self.layer55(out)                                                         # [BQ, c, h, w]

        out = self.res_aspp(out, history_mask)                                          # [BQ, c, h, w]
        output = self.layer7(out)                                                       # [BQ, 2, h, w]
        return output

    def res_aspp(self, features, history_mask):
        out = features
        _, _, h, w = features.size()
        history_mask = history_mask.view(-1, *history_mask.shape[-3:])

        if self.use_history:
            out_2 = torch.cat((out, history_mask), dim=1)                   # [BQ, c + 2, h, w]
        else:
            out_2 = out
        out = out + self.residual_1(out_2)                                  # [BQ, c, h, w]
        out = out + self.residual_2(out)                                    # [BQ, c, h, w]
        out = out + self.residual_3(out)                                    # [BQ, c, h, w]

        global_feat = F.avg_pool2d(out, (h, w))                             # [BQ, c, 1, 1]
        global_feat = self.aspp_0(global_feat)                              # [BQ, c, 1, 1]
        global_feat = global_feat.expand(-1, -1, h, w)                      # [BQ, c, h, w]
        out = torch.cat((global_feat,
                         self.aspp_1(out),
                         self.aspp_2(out),
                         self.aspp_3(out),
                         self.aspp_4(out)), dim=1)                          # [BQ, 5c, h, w]
        out = self.layer6(out)                                              # [BQ, c, h, w]
        return out

    def init_weights(self, pretrained):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)

        if pretrained is not None:
            self.encoder.init_weights(pretrained)

    @net_ingredient.capture
    def maybe_fix_params(self,
                         freeze_backbone):
        # A list of generators which are disposable
        variables = [self.encoder.conv1.parameters(),
                     self.encoder.layer1.parameters(),
                     self.encoder.layer2.parameters(),
                     self.encoder.layer3.parameters()]

        if freeze_backbone:
            for var_list in variables:
                for var in var_list:
                    var.requires_grad = False


ModelClass = CaNet
