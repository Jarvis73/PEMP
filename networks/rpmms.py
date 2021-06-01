import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sacred import Ingredient

from networks import backbones

net_ingredient = Ingredient("net", save_git_info=False)
pretrained_weights = {
    "resnet50": Path(__file__).parents[1] / "data/resnet50-19c8e357.pth",
}
backbone_error = "Not supported backbone '{}'. [resnet50]"


@net_ingredient.config
def net_config():
    dist_scalar = 20                        # int, a factor multiplied to cosine distance results
    init_channels = 3                       # int, input channels of the model
    out_channels = 512                      # int, output channels of the feature extractor
    backbone = "resnet50"                   # str, structure of the feature extractor. [vgg16, resnet50, resnet101]
    protos = 3                              # int, number of prototypes per class
    drop_rate = 0.5                         # float, drop rate used in the dropout


class PMMs(nn.Module):
    """
    Prototype Mixture Models
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    """

    def __init__(self, c, k=3, stage_num=10):
        super(PMMs, self).__init__()
        self.stage_num = stage_num
        self.num_pro = k
        mu = torch.Tensor(1, c, k).cuda()
        mu.normal_(0, math.sqrt(2. / k))  # Init mu
        self.mu = self._l2norm(mu, dim=1)
        self.kappa = 20
        # self.register_buffer('mu', mu)

    def forward(self, support_feature, support_mask, query_feature):
        prototypes, mu_f, mu_b = self.generate_prototype(support_feature, support_mask)
        Prob_map, P = self.discriminative_model(query_feature, mu_f, mu_b)

        return prototypes, Prob_map

    def _l2norm(self, inp, dim):
        """"Normlize the inp tensor with l2-norm.
        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.
        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        Returns:
            (tensor) The normalized tensor.
        """
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def EM(self, x):
        """
        EM method
        :param x: feauture  b * c * n
        :return: mu
        """
        b = x.shape[0]
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                # E STEP:
                z = self.Kernel(x, mu)
                z = F.softmax(z, dim=2)  # b * n * k
                # M STEP:
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)  # b * c * k

                mu = self._l2norm(mu, dim=1)

        mu = mu.permute(0, 2, 1)  # b * k * c

        return mu

    def Kernel(self, x, mu):
        x_t = x.permute(0, 2, 1)  # b * n * c
        z = self.kappa * torch.bmm(x_t, mu)  # b * n * k

        return z

    def get_prototype(self,x):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n
        mu = self.EM(x) # b * k * c

        return mu

    def generate_prototype(self, feature, mask):
        mask = F.interpolate(mask, feature.shape[-2:], mode='bilinear', align_corners=True)

        mask_bg = 1-mask

        # foreground
        z = mask * feature
        mu_f = self.get_prototype(z)
        mu_ = []
        for i in range(self.num_pro):
            mu_.append(mu_f[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))

        # background
        z_bg = mask_bg * feature
        mu_b = self.get_prototype(z_bg)

        return mu_, mu_f, mu_b

    def discriminative_model(self, query_feature, mu_f, mu_b):

        mu = torch.cat([mu_f, mu_b], dim=1)
        mu = mu.permute(0, 2, 1)

        b, c, h, w = query_feature.size()
        x = query_feature.view(b, c, h * w)  # b * c * n
        with torch.no_grad():

            x_t = x.permute(0, 2, 1)  # b * n * c
            z = torch.bmm(x_t, mu)  # b * n * k

            z = F.softmax(z, dim=2)  # b * n * k

        P = z.permute(0, 2, 1)

        P = P.view(b, self.num_pro * 2, h, w) # b * k * w * h  probability map
        P_f = torch.sum(P[:, 0:self.num_pro], dim=1).unsqueeze(dim=1) # foreground
        P_b = torch.sum(P[:, self.num_pro:], dim=1).unsqueeze(dim=1) # background

        Prob_map = torch.cat([P_b, P_f], dim=1)

        return Prob_map, P


class RPMMs(backbones.BaseModel, nn.Module):
    @net_ingredient.capture
    def __init__(self, logger,
                 init_channels, out_channels, backbone, drop_rate):

        self.inplanes = 64
        self.num_pro_list = [1, 3, 6]
        self.num_pro = self.num_pro_list[0]
        super(RPMMs, self).__init__()
        pretrained = pretrained_weights[backbone]

        self.model_res = backbones.ResNet(init_channels,
                                          backbones.BottleNeck,
                                          layers=[3, 4, 6],
                                          freeze_bn=True,
                                          ret_features=True,
                                          pretrained=pretrained)
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer55 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2,
                      bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=drop_rate),
        )

        self.layer56 = nn.Sequential(
            nn.Conv2d(in_channels=256+2, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1,
                      bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=drop_rate),
        )

        self.layer6 = backbones.ASPP(tail=False)

        self.layer7 = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=drop_rate),
        )

        self.layer9 = nn.Conv2d(256, 2, kernel_size=1, stride=1, bias=True) # numclass = 2

        self.residule1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256 + 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.residule2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.residule3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )

        logger.info(f"           ==> Model {self.__class__.__name__} created")

    def forward(self, sup_img, sup_mask, qry_img, out_shape=None):
        B, S, channel, H, W = sup_img.size()
        Q = qry_img.size(1)

        query_rgb = qry_img.view(B * Q, channel, H, W)
        support_rgb = sup_img.view(B * S, channel, H, W)
        support_mask = sup_mask.view(B * S, 2, H, W)[:, :1] # Only foreground mask

        # extract support feature
        support_feature = self.extract_feature_res(support_rgb)

        # extract query feature
        query_feature = self.extract_feature_res(query_rgb)

        feature_size = query_feature.shape[-2:]

        # feature concate
        Pseudo_mask = (torch.zeros(B, 2, 50, 50)).cuda()
        out_list = []
        for num in self.num_pro_list:
            self.num_pro = num
            self.PMMs = PMMs(256, num).cuda()
            vec_pos, Prob_map = self.PMMs(support_feature, support_mask, query_feature)

            for i in range(num):
                vec = vec_pos[i]
                exit_feat_in_ = self.f_v_concate(query_feature, vec, feature_size)
                exit_feat_in_ = self.layer55(exit_feat_in_)
                if i == 0:
                    exit_feat_in = exit_feat_in_
                else:
                    exit_feat_in = exit_feat_in + exit_feat_in_

            exit_feat_in = torch.cat([exit_feat_in, Prob_map], dim=1)
            exit_feat_in = self.layer56(exit_feat_in)

            # segmentation
            out, out_softmax = self.Segmentation(exit_feat_in, Pseudo_mask)
            Pseudo_mask = out_softmax
            out_list.append(out)

        return support_feature, out_list[0], out_list[1], out

    def extract_feature_res(self, rgb):
        out_resnet = self.model_res(rgb)
        stage2_out = out_resnet[1]
        stage3_out = out_resnet[2]
        out_23 = torch.cat([stage2_out, stage3_out], dim=1)
        feature = self.layer5(out_23)

        return feature

    def f_v_concate(self, feature, vec_pos, feature_size):
        fea_pos = vec_pos.expand(-1, -1, feature_size[0], feature_size[1])  # tile for cat
        exit_feat_in = torch.cat([feature, fea_pos], dim=1)

        return exit_feat_in

    def Segmentation(self, feature, history_mask):
        feature_size = feature.shape[-2:]

        history_mask = F.interpolate(history_mask, feature_size, mode='bilinear', align_corners=True)
        out = feature
        out_plus_history = torch.cat([feature, history_mask], dim=1)
        out = out + self.residule1(out_plus_history)
        out = out + self.residule2(out)
        out = out + self.residule3(out)

        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer9(out)

        out_softmax = F.softmax(out, dim=1)

        return out, out_softmax

    def get_loss(self, logits, query_label):
        bce_logits_func = nn.CrossEntropyLoss()
        support_feature, out0, out1, outB_side = logits

        b, c, h, w = query_label.size()
        out0 = F.interpolate(out0, size=(h, w), mode='bilinear', align_corners=True)
        out1 = F.interpolate(out1, size=(h, w), mode='bilinear', align_corners=True)
        outB_side = F.interpolate(outB_side, size=(h, w), mode='bilinear', align_corners=True)

        bb, cc, _, _ = outB_side.size()

        out0 = out0.view(b, cc, h * w)
        out1 = out1.view(b, cc, h * w)
        outB_side = outB_side.view(b, cc, h * w)
        query_label = query_label.view(b, -1)

        loss_bce_seg0 = bce_logits_func(out0, query_label.long())
        loss_bce_seg1 = bce_logits_func(out1, query_label.long())
        loss_bce_seg2 = bce_logits_func(outB_side, query_label.long())

        loss = loss_bce_seg0+loss_bce_seg1+loss_bce_seg2

        return loss, loss_bce_seg2, loss_bce_seg1

    def get_pred(self, logits, query_image):
        outB, outA_pos, outB_side1, outB_side = logits
        w, h = query_image.size()[-2:]
        outB_side = F.interpolate(outB_side, size=(w, h), mode='bilinear', align_corners=True)
        out_softmax = F.softmax(outB_side, dim=1)
        values, pred = torch.max(out_softmax, dim=1)
        return out_softmax, pred

    def load_weights(self, ckpt_path, logger):
        weights = torch.load(str(ckpt_path), map_location='cpu')
        if "state_dict" in weights:
            weights = weights["state_dict"]

        try:
            self.load_state_dict(weights)
        except RuntimeError as e:
            pre_keys = list(weights.keys())
            cur_weights = self.state_dict()
            cur_keys = list(cur_weights.keys())

            for key in cur_keys:
                old_key = key.replace("aspp", "layer6")
                if old_key in pre_keys:
                    cur_weights[key] = weights[old_key]
                else:
                    print("Checkpoint:", str(ckpt_path))
                    raise e
            self.load_state_dict(cur_weights)

        # Print short path if possible
        try:
            short_path = Path(ckpt_path).relative_to(Path(__file__).parents[1])
        except ValueError:
            short_path = ckpt_path
        logger.info(f"           ==> Model {self.__class__.__name__} initialized from {short_path}")


ModelClass = RPMMs
