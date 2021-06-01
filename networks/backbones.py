from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D

affine = True
__all__ = [
    "BaseModel",
    "BottleNeck",
    "ResNet",
    "ResNetCM",
    "ASPP",
    "ASPPV2",
    "VGG16",
    "VGG16CM"
]


class BaseModel(nn.Module):
    def load_weights(self, ckpt_path, logger):
        weights = torch.load(str(ckpt_path), map_location='cpu')
        if "state_dict" in weights:
            weights = weights["state_dict"]
        self.load_state_dict(weights)

        # Print short path if possible
        try:
            short_path = Path(ckpt_path).relative_to(Path(__file__).parents[1])
        except ValueError:
            short_path = ckpt_path
        logger.info(f"           ==> Model {self.__class__.__name__} initialized from {short_path}")

    def maybe_fix_params(self, fix=False):
        if fix:
            # Fix parameters when training PEMP_Stage2
            for var in self.parameters():
                var.requires_grad = False


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, freeze_bn=False):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if freeze_bn:
            for var in self.bn1.parameters():
                var.requires_grad = False
            for var in self.bn2.parameters():
                var.requires_grad = False
            for var in self.bn3.parameters():
                var.requires_grad = False

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet-50 variant
    """
    def __init__(self, init_c, block, layers, freeze_bn=False, ret_features=True, pretrained=None):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.ret_features = ret_features

        self.conv1 = nn.Conv2d(init_c, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        if freeze_bn:
            for var in self.bn1.parameters():
                var.requires_grad = False

        self.layer1 = self._make_layer(block, 64, layers[0], freeze_bn=freeze_bn)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, freeze_bn=freeze_bn)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, freeze_bn=freeze_bn)
        if len(layers) > 3:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, freeze_bn=freeze_bn)

        if pretrained is not None:
            self.init_weights(pretrained)

    def _make_layer(self, block: BottleNeck, planes, blocks, stride=1, dilation=1, freeze_bn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine)
            )
            if freeze_bn:
                for var in downsample._modules['1'].parameters():
                    var.requires_grad = False

        layers = [block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.max_pool(self.relu(self.bn1(self.conv1(x))))
        l1 = self.layer1(out)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        if hasattr(self, "layer4"):
            l4 = self.layer4(l3)
            if self.ret_features:
                return l1, l2, l3, l4
            return l4
        if self.ret_features:
            return l1, l2, l3
        return l3

    def init_weights(self, pretrained):
        pre_weights = torch.load(pretrained, map_location='cpu')
        pre_keys = list(pre_weights.keys())
        cur_weights = self.state_dict()
        cur_keys = list(cur_weights.keys())

        if not hasattr(self, "layer4"):
            for key in pre_keys:
                if key.split(".")[0] != "layer4":
                    cur_weights[key] = pre_weights[key]
                else:
                    break
        else:
            for key in pre_keys:
                if key.split(".")[0] != "fc":
                    cur_weights[key] = pre_weights[key]
                else:
                    break

        self.load_state_dict(cur_weights)


class ResNetCM(nn.Module):
    """
    ResNet-50 variant with communication modules
    """
    def __init__(self, init_c, block, layers, freeze_bn=False, ret_features=True, pretrained=None, shot_query=None):
        super(ResNetCM, self).__init__()
        self.inplanes = 64
        self.ret_features = ret_features
        self.spq = shot_query
        n = 2   # Number channels of the CM output

        self.conv1 = nn.Conv2d(init_c, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        if freeze_bn:
            for var in self.bn1.parameters():
                var.requires_grad = False

        self.layer1 = self._make_layer(block, 64, layers[0], freeze_bn=freeze_bn, n=n)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, freeze_bn=freeze_bn, n=n)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, freeze_bn=freeze_bn, n=n)

        self.linear1 = nn.Linear(2 * 64, n, bias=True)
        self.linear2 = nn.Linear(2 * 256, n, bias=True)
        self.linear3 = nn.Linear(2 * 512, n, bias=True)

        if pretrained is not None:
            self.init_weights(pretrained, n)

    def _make_layer(self, block: BottleNeck, planes, blocks, stride=1, dilation=1, freeze_bn=False, n=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes + n, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine)
            )
            if freeze_bn:
                for var in downsample._modules['1'].parameters():
                    var.requires_grad = False

        layers = [block(self.inplanes + n, planes, stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def comm(self, x, mask, linear, stride=2):
        mask = F.max_pool2d(mask, 3, stride, 1)                                     # [B * (S + Q), 1, h, w]
        masked_x = (x * mask).view(*x.shape[:2], -1)                                # [B(S + Q), c, hw]
        mask_flat = mask.view(x.shape[0], 1, -1)                                    # [B * (s + Q), 1, hw]
        mean = masked_x.mean(dim=-1)                                                # [B(S + Q), c]
        mean = mean.view(x.shape[0] // self.spq, self.spq, -1).mean(dim=1)          # [B, c]

        max_ = masked_x.max(dim=-1)[0]                                              # [B(S + Q), c]
        max_ = max_.view(x.shape[0] // self.spq, self.spq, -1).mean(dim=1)          # [B, c]
        feat = torch.cat([mean, max_], dim=1)                                       # [B, 2c]
        feat = linear(feat)                                                         # [B, n]
        feat = feat.unsqueeze(dim=1).unsqueeze(dim=-1).unsqueeze(dim=-1)\
            .expand(-1, self.spq, -1, *x.shape[-2:])                                # [B, S + Q, n, h, w]
        feat = feat.reshape(x.shape[0], -1, *x.shape[-2:])                          # [B(S + Q), n, h, w]
        return feat, mask

    def forward(self, x):
        features = []
        x, mask = x
        mask = F.max_pool2d(mask, 3, 2, 1)
        x1 = self.max_pool(self.relu(self.bn1(self.conv1(x))))

        ci1, mask = self.comm(x1, mask, self.linear1)
        x1 = torch.cat([x1, ci1], dim=1)
        x2 = self.layer1(x1)
        features.append(x2)

        ci2, mask = self.comm(x2, mask, self.linear2, stride=1)
        x2 = torch.cat([x2, ci2], dim=1)
        x3 = self.layer2(x2)
        features.append(x3)

        ci3, mask = self.comm(x3, mask, self.linear3)
        x3 = torch.cat([x3, ci3], dim=1)
        x4 = self.layer3(x3)
        features.append(x4)

        if self.ret_features:
            return features
        return x4

    def init_weights(self, pretrained, n):
        pre_weights = torch.load(pretrained, map_location='cpu')
        pre_keys = list(pre_weights.keys())
        cur_weights = self.state_dict()
        cur_keys = list(cur_weights.keys())

        for i, key in enumerate(pre_keys):
            temp = pre_weights[key]
            if i == 0:
                zeros = torch.zeros((64, 1, 7, 7), dtype=temp.dtype)
                temp = torch.cat((temp, zeros), dim=1)
            elif "layer4" in key:
                break
            elif "downsample.0.weight" in key or "0.conv1.weight" in key:
                c1 = True if "downsample.0.weight" in key else False
                if "layer1" in key:
                    inc = 256 if c1 else 64
                elif "layer2" in key:
                    inc = 512 if c1 else 128
                elif "layer3" in key:
                    inc = 1024 if c1 else 256
                else:
                    raise ValueError(key)
                zeros = torch.zeros((inc, n, 1, 1), dtype=temp.dtype)
                temp = torch.cat((temp, zeros), dim=1)
            cur_weights[key] = temp

        self.load_state_dict(cur_weights)


class ASPP(nn.Module):
    def __init__(self, inc=256, midc=256, outc=512, drop_rate=0.5, tail=True):
        super(ASPP, self).__init__()
        self.aspp_0 = nn.Sequential(
            nn.Conv2d(inc, midc, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(drop_rate)
        )
        self.aspp_1 = nn.Sequential(
            nn.Conv2d(inc, midc, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(drop_rate)
        )
        self.aspp_2 = nn.Sequential(
            nn.Conv2d(inc, midc, kernel_size=3, stride=1, padding=6, dilation=6, bias=True),
            nn.ReLU(),
            nn.Dropout2d(drop_rate)
        )
        self.aspp_3 = nn.Sequential(
            nn.Conv2d(inc, midc, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(),
            nn.Dropout2d(drop_rate)
        )
        self.aspp_4 = nn.Sequential(
            nn.Conv2d(inc, midc, kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(),
            nn.Dropout2d(drop_rate)
        )
        if tail:
            self.layer6 = nn.Conv2d(midc * 5, outc, kernel_size=1, bias=True)

    def forward(self, x):
        global_feat = F.adaptive_avg_pool2d(x, (1, 1))  # [bs * query, c, 1, 1]
        global_feat = self.aspp_0(global_feat)  # [bs * query, c, 1, 1]
        global_feat = global_feat.expand(-1, -1, *x.shape[-2:])  # [bs * query, c, h, w]
        out = torch.cat((global_feat,
                         self.aspp_1(x),
                         self.aspp_2(x),
                         self.aspp_3(x),
                         self.aspp_4(x)), dim=1)  # [bs * query, 5c, h, w]
        if hasattr(self, "layer6"):
            out = self.layer6(out)
        return out


class ASPPV2(nn.Module):
    def __init__(self, inc=256, midc=256, outc=512, drop_rate=0.5, block_size=4):
        super(ASPPV2, self).__init__()
        self.aspp_0 = nn.Sequential(
            nn.BatchNorm2d(inc),
            DropBlock2D(drop_prob=drop_rate, block_size=block_size),
            nn.Conv2d(inc, midc, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
        )
        self.aspp_1 = nn.Sequential(
            nn.BatchNorm2d(inc),
            DropBlock2D(drop_prob=drop_rate, block_size=block_size),
            nn.Conv2d(inc, midc, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
        )
        self.aspp_2 = nn.Sequential(
            nn.BatchNorm2d(inc),
            DropBlock2D(drop_prob=drop_rate, block_size=block_size),
            nn.Conv2d(inc, midc, kernel_size=3, stride=1, padding=6, dilation=6, bias=True),
            nn.ReLU(),
        )
        self.aspp_3 = nn.Sequential(
            nn.BatchNorm2d(inc),
            DropBlock2D(drop_prob=drop_rate, block_size=block_size),
            nn.Conv2d(inc, midc, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(),
        )
        self.aspp_4 = nn.Sequential(
            nn.BatchNorm2d(inc),
            DropBlock2D(drop_prob=drop_rate, block_size=block_size),
            nn.Conv2d(inc, midc, kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(),
        )
        self.layer6 = nn.Conv2d(midc * 5, outc, kernel_size=1, bias=True)

    def forward(self, x):
        global_feat = F.adaptive_avg_pool2d(x, (1, 1))  # [bs * query, c, 1, 1]
        global_feat = self.aspp_0(global_feat)  # [bs * query, c, 1, 1]
        global_feat = global_feat.expand(-1, -1, *x.shape[-2:])  # [bs * query, c, h, w]
        out = torch.cat((global_feat,
                         self.aspp_1(x),
                         self.aspp_2(x),
                         self.aspp_3(x),
                         self.aspp_4(x)), dim=1)  # [bs * query, 5c, h, w]
        out = self.layer6(out)
        return out


class VGG16(nn.Module):
    def __init__(self, init=3, pretrained=None, lastRelu=False):
        super(VGG16, self).__init__()
        layers = [
            nn.Conv2d(init, 64, kernel_size=3, padding=1),      nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),        nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),       nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),      nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),      nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),      nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),      nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),      nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),      nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),      nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),   # stride = 1

            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),  nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),  nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
        ]
        if lastRelu:
            layers.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*layers)

        self.init_weights(pretrained)

    def forward(self, x):
        return self.features(x)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        if pretrained:
            pre_weights = torch.load(pretrained, map_location='cpu')
            pre_keys = list(pre_weights.keys())
            cur_weights = self.state_dict()
            cur_keys = list(cur_weights.keys())

            for i in range(26):
                cur_weights[cur_keys[i]] = pre_weights[pre_keys[i]]

            self.load_state_dict(cur_weights)


class VGG16CM(nn.Module):
    def __init__(self, init=4, pretrained=None, lastRelu=False, shot_query=None):
        """ VGG16 with Communication Module """
        self.spq = shot_query
        n = 2   # Number channels of the CM output

        super(VGG16CM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(init, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64 + n, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128 + n, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256 + n, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # stride = 1
        )
        layers = [
            nn.Conv2d(512 + n, 512, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
        ]
        if lastRelu:
            layers.append(nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(*layers)

        self.linear1 = nn.Linear(n * 64, n, bias=True)
        self.linear2 = nn.Linear(n * 128, n, bias=True)
        self.linear3 = nn.Linear(n * 256, n, bias=True)
        self.linear4 = nn.Linear(n * 512, n, bias=True)

        self.init_weights(pretrained, n)

    def comm(self, x, mask, linear, stride=2):
        mask = F.max_pool2d(mask, 3, stride, 1)  # [bs * (shot + query), 1, h, w]
        masked_x = (x * mask).view(x.shape[0] // self.spq, self.spq, x.shape[1], -1)
        mean = masked_x.mean(dim=-1).mean(dim=1)  # [bs, c]
        max_ = masked_x.max(dim=-1)[0].mean(dim=1)  # [bs, c]
        feat = torch.cat([mean, max_], dim=1)
        feat = linear(feat)
        feat = feat.unsqueeze(dim=1).unsqueeze(dim=-1).unsqueeze(dim=-1)\
            .expand(-1, self.spq, -1, *x.shape[-2:])
        feat = feat.reshape(x.shape[0], -1, *x.shape[-2:])
        return feat, mask

    def forward(self, x):
        x, mask = x
        x1 = self.layer1(x)
        ci1, mask = self.comm(x1, mask, self.linear1)
        x1 = torch.cat([x1, ci1], dim=1)

        x2 = self.layer2(x1)
        ci2, mask = self.comm(x2, mask, self.linear2)
        x2 = torch.cat([x2, ci2], dim=1)

        x3 = self.layer3(x2)
        ci3, mask = self.comm(x3, mask, self.linear3)
        x3 = torch.cat([x3, ci3], dim=1)

        x4 = self.layer4(x3)
        ci4, mask = self.comm(x4, mask, self.linear4, stride=1)
        x4 = torch.cat([x4, ci4], dim=1)

        x5 = self.layer5(x4)
        return x5

    def init_weights(self, pretrained=None, n=2):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        if pretrained is not None:
            pre_weights = torch.load(pretrained, map_location='cpu')
            pre_keys = list(pre_weights.keys())
            cur_weights = self.state_dict()
            cur_keys = list(cur_weights.keys())

            for i in range(26):
                temp = pre_weights[pre_keys[i]]
                if i == 0:
                    zeros = torch.zeros((64, 1, 3, 3), dtype=temp.dtype)
                    temp = torch.cat((temp, zeros), dim=1)
                if self.cm:
                    if i == 4:
                        zeros = torch.zeros((128, n, 3, 3), dtype=temp.dtype)
                        temp = torch.cat((temp, zeros), dim=1)
                    elif i == 8:
                        zeros = torch.zeros((256, n, 3, 3), dtype=temp.dtype)
                        temp = torch.cat((temp, zeros), dim=1)
                    elif i == 14:
                        zeros = torch.zeros((512, n, 3, 3), dtype=temp.dtype)
                        temp = torch.cat((temp, zeros), dim=1)
                    elif i == 20:
                        zeros = torch.zeros((512, n, 3, 3), dtype=temp.dtype)
                        temp = torch.cat((temp, zeros), dim=1)
                cur_weights[cur_keys[i]] = temp

            self.load_state_dict(cur_weights)
